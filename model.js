// LA FRIES A.I. - client-side inference engine.
// A plain-JS port of the trained GRU's forward pass (model/gru.py's
// step_generate). No server, no API calls - this runs entirely in the
// browser using the exported weights in model_data.json.

class LaFries {
  constructor(data) {
    this.V = data.V;
    this.E = data.E;
    this.H = data.H;
    this.stoi = data.stoi;
    this.itos = data.itos;
    const p = data.params;
    this.Wemb = p.Wemb;       // [V][E]
    this.Wzr_x = p.Wzr_x;     // [E][2H]
    this.Wzr_h = p.Wzr_h;     // [H][2H]
    this.bzr = p.bzr;         // [2H]
    this.Wh_x = p.Wh_x;       // [E][H]
    this.Wh_h = p.Wh_h;       // [H][H]
    this.bh = p.bh;           // [H]
    this.Wy = p.Wy;           // [H][V]
    this.by = p.by;           // [V]
  }

  static sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  // x: array of length E, W: [rowsxcols] as array-of-arrays with `rows` rows,
  // returns array of length cols. (row-major: W[i][j])
  static vecMatAdd(x, W, cols, out) {
    const rows = x.length;
    for (let j = 0; j < cols; j++) out[j] = 0;
    for (let i = 0; i < rows; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const Wi = W[i];
      for (let j = 0; j < cols; j++) out[j] += xi * Wi[j];
    }
    return out;
  }

  step(charId, h) {
    const { E, H, V } = this;
    const xt = this.Wemb[charId];

    const zrPre = new Float64Array(2 * H);
    LaFries.vecMatAdd(xt, this.Wzr_x, 2 * H, zrPre);
    const hzr = new Float64Array(2 * H);
    LaFries.vecMatAdd(h, this.Wzr_h, 2 * H, hzr);
    const z = new Float64Array(H);
    const r = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      z[j] = LaFries.sigmoid(zrPre[j] + hzr[j] + this.bzr[j]);
      r[j] = LaFries.sigmoid(zrPre[H + j] + hzr[H + j] + this.bzr[H + j]);
    }

    const rh = new Float64Array(H);
    for (let j = 0; j < H; j++) rh[j] = r[j] * h[j];

    const preH1 = new Float64Array(H);
    LaFries.vecMatAdd(xt, this.Wh_x, H, preH1);
    const preH2 = new Float64Array(H);
    LaFries.vecMatAdd(rh, this.Wh_h, H, preH2);

    const hTilde = new Float64Array(H);
    const hNew = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      hTilde[j] = Math.tanh(preH1[j] + preH2[j] + this.bh[j]);
      hNew[j] = (1 - z[j]) * h[j] + z[j] * hTilde[j];
    }

    const logits = new Float64Array(V);
    LaFries.vecMatAdd(hNew, this.Wy, V, logits);
    for (let j = 0; j < V; j++) logits[j] += this.by[j];

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
