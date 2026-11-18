import { self as sw } from '$app/service-worker';
import { assets, immutable } from '$app/manifest';
import { version } from '$app/env';
import {
	DATA_CACHE,
	OFFLINE_MISS,
	SHELL,
	dataKey,
	dataUrlOf,
	precachedAssets,
	predownload,
	uninstall,
	type WorkerMessage
} from '#lib/offline.js';
import { openPush, showPush } from '#lib/pushFallback.js';

// 整頁載入（啟動、重新整理）時向網路要頁面，等這麼久還沒拿到就改用存好的空殼頁
const NAVIGATION_TIMEOUT_MS = 3000;
const PRECACHE = `precache-${version}`;
// 不對應實際檔案；放在資料快取裡，快取被清掉時記號一起消失
const MARKER = '__offline/predownload.json';
let running: Promise<boolean> | undefined;

// 建置時 VITE_SW_REMOVE=1：同一個網址換成自我移除版，裝置退回純網站（打包時另一邊的分支整段拿掉）
const removing = import.meta.env.VITE_SW_REMOVE === '1';

if (removing) {
	// 不等舊版放手；不掛 fetch，還被它掛著的分頁所有請求直接走網路
	sw.addEventListener('install', (event) => {
		event.waitUntil(sw.skipWaiting());
	});

	sw.addEventListener('activate', (event) => {
		event.waitUntil(uninstall(caches, sw.registration));
	});
} else {
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
		if (new URL(request.url).pathname.endsWith('/__data.json')) return event.respondWith(data(event));
		event.respondWith(precached(request));
	});

	sw.addEventListener('push', (event) => {
		event.waitUntil(showPush(sw.registration, event.data));
	});

	sw.addEventListener('notificationclick', (event) => {
		event.notification.close();
		event.waitUntil(openPush(sw.clients, event.notification.data, sw.registration.scope));
	});
}

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
async function data(event: FetchEvent) {
	const { request } = event;
	try {
		const res = await fetch(request);
		if (res.ok) event.waitUntil(saveData(request.url, res.clone()));
		return res;
	} catch {
		const saved = await (await caches.open(DATA_CACHE)).match(dataKey(request.url), { ignoreVary: true });
		if (saved) return saved;
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

async function saveData(url: string, res: Response) {
	await (await caches.open(DATA_CACHE)).put(dataKey(url), res);
}

if (!removing) {
	sw.addEventListener('message', (event) => {
		const message = event.data as WorkerMessage | null;
		if (message?.type === 'save') return event.waitUntil(save(message.path));
		if (message?.type === 'sync') event.waitUntil(sync(message.paths, event.ports[0]));
		// install 不 skipWaiting：新版停在等待，等使用者按更新條（或獨立視窗離開太久）才接手
		if (message?.type === 'activate') event.waitUntil(sw.skipWaiting());
	});
}

// 只接受字串、且在 scope 範圍內的路徑
function dataUrls(paths: unknown[]): string[] {
	return paths.flatMap((path) => {
		if (typeof path !== 'string') return [];
		const url = new URL(dataUrlOf(path), sw.registration.scope);
		return url.href.startsWith(sw.registration.scope) ? [url.href] : [];
	});
}

async function save(path: unknown) {
	const [url] = dataUrls([path]);
	if (!url) return;
	try {
		const res = await fetch(url, { cache: 'no-cache' });
		if (res.ok) await saveData(url, res);
	} catch {}
}

async function sync(paths: unknown, port: MessagePort | undefined) {
	let ready = false;
	try {
		if (!(await fromPrecache(SHELL))) await precache();
		ready = true;
		if (Array.isArray(paths)) ready = await predownloadOnce(dataUrls(paths));
	} catch {
		ready = false;
	}
	port?.postMessage({ ready });
}

function predownloadOnce(urls: string[]) {
	running ??= caches
		.open(DATA_CACHE)
		.then((store) =>
			predownload({ store, fetch, urls, version, markerUrl: new URL(MARKER, sw.registration.scope).href })
		)
		.finally(() => (running = undefined));
	return running;
}
