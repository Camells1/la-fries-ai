// LA FRIES A.I. - client-side inference engine.
// A plain-JS port of the trained GRU's forward pass (model/gru.py's
// step_generate). No server, no API calls - this runs entirely in the
// browser using binary weight shards described by a small JSON manifest.

function base64ToBytes(b64) {
  const binStr = atob(b64);
  const bytes = new Uint8Array(binStr.length);
  for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
  return bytes;
}

async function loadLaFriesModel(manifestUrl) {
  const res = await fetch(manifestUrl);
  const manifest = await res.json();

  // shards are base64 text (not raw binary) -- the Artifact hosting
  // platform this also deploys to only serves a fixed extension allowlist
  const byteArrays = await Promise.all(
    manifest.shards.map(async (name) => {
      const url = new URL(name, new URL(manifestUrl, location.href)).toString();
      const r = await fetch(url);
      const text = await r.text();
      return base64ToBytes(text);
    })
  );

  const flat = new Uint8Array(manifest.total_bytes);
  let pos = 0;
  for (const bytes of byteArrays) {
    flat.set(bytes, pos);
    pos += bytes.length;
  }
  const flatFloats = new Float32Array(flat.buffer);

  const params = {};
  for (const [name, info] of Object.entries(manifest.params)) {
    const count = info.shape.reduce((a, b) => a * b, 1);
    const start = info.offset / 4; // byte offset -> float32 index
    params[name] = { data: flatFloats.subarray(start, start + count), shape: info.shape };
  }

  return new LaFries(manifest.V, manifest.E, manifest.H, params, manifest.stoi, manifest.itos);
}

class LaFries {
  constructor(V, E, H, params, stoi, itos) {
    this.V = V; this.E = E; this.H = H;
    this.stoi = stoi; this.itos = itos;
    this.p = params; // {name: {data: Float32Array, shape: [...]}}
  }

  param_count() {
    const { V, E, H } = this;
    return V * E + 3 * (E * H + H * H + H) + H * V + V;
  }

  static sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  // x: flat array-like of length `rows`; W: {data, shape:[rows,cols]} row-major.
  // Accumulates x @ W into `out` (length cols).
  static vecMatAdd(x, W, out) {
    const data = W.data;
    const cols = W.shape[1];
    const rows = W.shape[0];
    for (let j = 0; j < cols; j++) out[j] = 0;
    for (let i = 0; i < rows; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const base = i * cols;
      for (let j = 0; j < cols; j++) out[j] += xi * data[base + j];
    }
    return out;
  }

  step(charId, h) {
    const { E, H, V, p } = this;
    const embData = p.Wemb.data;
    const xt = embData.subarray(charId * E, charId * E + E);

    const zrPre = new Float64Array(2 * H);
    LaFries.vecMatAdd(xt, p.Wzr_x, zrPre);
    const hzr = new Float64Array(2 * H);
    LaFries.vecMatAdd(h, p.Wzr_h, hzr);
    const z = new Float64Array(H);
    const r = new Float64Array(H);
    const bzr = p.bzr.data;
    for (let j = 0; j < H; j++) {
      z[j] = LaFries.sigmoid(zrPre[j] + hzr[j] + bzr[j]);
      r[j] = LaFries.sigmoid(zrPre[H + j] + hzr[H + j] + bzr[H + j]);
    }

    const rh = new Float64Array(H);
    for (let j = 0; j < H; j++) rh[j] = r[j] * h[j];

    const preH1 = new Float64Array(H);
    LaFries.vecMatAdd(xt, p.Wh_x, preH1);
    const preH2 = new Float64Array(H);
    LaFries.vecMatAdd(rh, p.Wh_h, preH2);

    const bh = p.bh.data;
    const hTilde = new Float64Array(H);
    const hNew = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      hTilde[j] = Math.tanh(preH1[j] + preH2[j] + bh[j]);
      hNew[j] = (1 - z[j]) * h[j] + z[j] * hTilde[j];
    }

    const logits = new Float64Array(V);
    LaFries.vecMatAdd(hNew, p.Wy, logits);
    const by = p.by.data;
    for (let j = 0; j < V; j++) logits[j] += by[j];

    return { logits, h: hNew };
  }

  sampleFromLogits(logits, temperature) {
    const V = logits.length;
    const scaled = new Float64Array(V);
    let max = -Infinity;
    for (let i = 0; i < V; i++) {
      scaled[i] = logits[i] / Math.max(temperature, 1e-6);
      if (scaled[i] > max) max = scaled[i];
    }
    let sum = 0;
    for (let i = 0; i < V; i++) {
      scaled[i] = Math.exp(scaled[i] - max);
      sum += scaled[i];
    }
    let r = Math.random() * sum;
    for (let i = 0; i < V; i++) {
      r -= scaled[i];
      if (r <= 0) return i;
    }
    return V - 1;
  }

  // Generates a reply to `promptText`, formatted the same way training was.
  // `history` (prior "### Prompt: ...\n### Response: ...\n" turns) is
  // prepended so it can carry a real conversation, not just one-shot Q&A.
  *generate(promptText, { maxChars = 500, temperature = 0.7, end = "~~~", history = "" } = {}) {
    let h = new Float64Array(this.H);
    const full = `${history}### Prompt: ${promptText}\n### Response:`;
    for (const ch of full) {
      const id = this.stoi[ch] !== undefined ? this.stoi[ch] : this.stoi[" "];
      const { h: hNew } = this.step(id, h);
      h = hNew;
    }

    let lastId = this.stoi[":"] !== undefined ? this.stoi[":"] : 0;
    let tail = "";
    for (let i = 0; i < maxChars; i++) {
      const { logits, h: hNew } = this.step(lastId, h);
      h = hNew;
      const nextId = this.sampleFromLogits(logits, temperature);
      const ch = this.itos[String(nextId)];
      tail += ch;
      if (tail.endsWith(end)) {
        yield { ch: null, done: true, text: tail.slice(0, -end.length) };
        return;
      }
      yield { ch, done: false, text: tail };
      lastId = nextId;
    }
    yield { ch: null, done: true, text: tail };
  }
}
