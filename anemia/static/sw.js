const CACHE_NAME = 'ocula-ml-cache-v1';
const urlsToCache = [
  '/',
  '/detect',
  '/evaluation',
  '/static/css/style.css',
  '/static/js/camera.js',
  '/manifest.json'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Membuka cache PWA OculaML');
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', event => {
  // Hanya simpan dan pakai fetch jika permintaan berformat GET
  if (event.request.method !== 'GET') return;
  
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Kembalikan versi cache jika ada
        if (response) {
          return response;
        }
        return fetch(event.request);
      })
  );
});
