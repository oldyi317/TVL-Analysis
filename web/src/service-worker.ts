import { self as sw } from '$app/service-worker';
import { assets, immutable } from '$app/manifest';
import { version } from '$app/env';
import { OFFLINE_MISS, SHELL, precachedAssets } from '#lib/offline.js';

// 整頁載入（啟動、重新整理）時向網路要頁面，等這麼久還沒拿到就改用存好的空殼頁
const NAVIGATION_TIMEOUT_MS = 3000;
const PRECACHE = `precache-${version}`;

sw.addEventListener('install', (event) => {
	event.waitUntil(precache());
});

sw.addEventListener('activate', (event) => {
	event.waitUntil(dropOldPrecaches().then(() => sw.clients.claim()));
});

sw.addEventListener('fetch', (event) => {
	const { request } = event;
	if (request.method !== 'GET' || !request.url.startsWith(sw.registration.scope)) return;
	if (request.mode === 'navigate') return event.respondWith(page(request));
	if (new URL(request.url).pathname.endsWith('/__data.json')) return event.respondWith(data(request));
	event.respondWith(precached(request));
});

async function precache() {
	const cache = await caches.open(PRECACHE);
	// 空殼頁與靜態檔的檔名固定，要向伺服器重新驗證；否則可能存到上一版的空殼頁，它指向的程式檔新版沒有
	const fixed = [SHELL, ...precachedAssets(assets.map((a) => a.path))].map(
		(path) => new Request(path, { cache: 'no-cache' })
	);
	await cache.addAll([...immutable.map((a) => a.path), ...fixed]);
}

async function dropOldPrecaches() {
	for (const key of await caches.keys()) {
		if (key.startsWith('precache-') && key !== PRECACHE) await caches.delete(key);
	}
}

async function page(request: Request) {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), NAVIGATION_TIMEOUT_MS);
	try {
		return await fetch(request, { signal: controller.signal });
	} catch {
		return (await fromPrecache(SHELL)) ?? Response.error();
	} finally {
		clearTimeout(timer);
	}
}

// 做法成立的前提：根 layout 沒有 server load（沒有 src/routes/+layout.server.ts），
// 所以資料檔抓不到時 Kit 只畫根層的 +error.svelte。
// 之後若替根 layout 加了 server load，Kit 會再抓一次根層資料、失敗就整頁重新載入，要重驗離線說明頁。
async function data(request: Request) {
	try {
		return await fetch(request);
	} catch {
		return new Response(JSON.stringify({ message: OFFLINE_MISS }), {
			status: 503,
			headers: { 'content-type': 'application/json' }
		});
	}
}

async function precached(request: Request) {
	return (await fromPrecache(request)) ?? fetch(request);
}

// 預存的檔不隨請求標頭而變；伺服器回 Vary 時（vite preview 回 Vary: Origin），頁面的請求會對不上預存時的請求，所以忽略 Vary
function fromPrecache(request: RequestInfo) {
	return caches.match(request, { cacheName: PRECACHE, ignoreVary: true });
}
