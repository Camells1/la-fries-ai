// La Peace AI - service worker for offline / installable use on Chrome OS.
// Bump CACHE_NAME on any real content change so old caches get cleared.
const CACHE_NAME = "la-fries-ai-v5";
const CORE_ASSETS = [
  "./",
  "./index.html",
  "./model.js",
  "./manifest.json",
  "./icon-180.png",
  "./icon-192.png",
  "./icon-512.png",
  "./icon-512-maskable.png",
];

// Only the big model-weight shards are worth being cache-first about (large,
// rarely change, genuinely benefit from not re-downloading). Everything else
// -- especially model.js and index.html -- must be network-first, or a
// returning visitor can get stuck on stale code indefinitely after an
// update (this bit us during testing: a cached model.js referenced a
// function that no longer matched a redeployed page).
function isModelShard(url) {
  return /model_data_.*\.(txt|json)$/.test(url) || /\.bin$/.test(url);
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  if (isModelShard(req.url)) {
    // cache-first: large, rarely change, worth not re-downloading
    event.respondWith(
      caches.match(req).then(
        (cached) =>
          cached ||
          fetch(req).then((res) => {
            const copy = res.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
            return res;
          })
      )
    );
  } else {
    // network-first for everything else (page, model.js, manifest, icons)
    // so a redeploy reaches returning visitors instead of getting stuck
    // behind a stale cached script indefinitely.
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
          return res;
        })
        .catch(() => caches.match(req).then((res) => res || caches.match("./index.html")))
    );
  }
});
