// 原型 service worker：手寫，不用 Workbox。設計見 prototypes/offline/README.md
import { self as sw } from '$app/service-worker';
import { immutable, assets } from '$app/manifest';
import { version } from '$app/env';

const BASE = new URL(sw.registration.scope).pathname.replace(/\/$/, '');
const PRE = `precache-${version}`;
const DATA = 'data-v1';
const CFG = 'proto-config';
const SHELL = `${BASE}/shell.html`;

type Cfg = { navTimeout: number; delay: number; offline: boolean; threshold: number; standalone: boolean };
const DEFAULTS: Cfg = { navTimeout: 3000, delay: 0, offline: false, threshold: 30 * 60 * 1000, standalone: false };
type Nav = { how: string; ms: number; url: string; at: number };
type Predl = { version: string; done: number; total: number; failed: number; running: boolean };
type State = { nav?: Nav; predl?: Predl };

const cfgKey = (name: string) => new Request(`${sw.location.origin}${BASE}/__proto/${name}`);

async function readJson<T>(name: string): Promise<T | undefined> {
	const cache = await caches.open(CFG);
	const hit = await cache.match(cfgKey(name));
	return hit ? ((await hit.json()) as T) : undefined;
}

async function writeJson(name: string, value: unknown) {
	const cache = await caches.open(CFG);
	await cache.put(cfgKey(name), new Response(JSON.stringify(value), { headers: { 'content-type': 'application/json' } }));
}

const getCfg = async (): Promise<Cfg> => ({ ...DEFAULTS, ...(await readJson<Partial<Cfg>>('config')) });
const getState = async (): Promise<State> => (await readJson<State>('state')) ?? {};
async function patchState(patch: Partial<State>) {
	await writeJson('state', { ...(await getState()), ...patch });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function fetchNet(input: Request | string, cfg: Cfg, init?: RequestInit) {
	if (cfg.offline) return Promise.reject(new TypeError('模擬離線'));
	return fetch(input, init);
}

// 回給導覽的 Response 不能是 redirected
async function clean(res: Response) {
	if (!res.redirected) return res;
	return new Response(await res.blob(), { status: res.status, statusText: res.statusText, headers: res.headers });
}

const precacheUrls = (() => {
	const skip = /\/(robots\.txt|og-image\.(png|svg))$/;
	const full = (path: string) => `${BASE}/${path.replace(/^\//, '')}`;
	const code = immutable.map((a) => full(a.path));
	const statics = assets.map((a) => full(a.path)).filter((p) => !skip.test(p));
	return { code, statics, shell: [SHELL] };
})();

async function pool<T>(items: T[], size: number, fn: (item: T) => Promise<void>) {
	let i = 0;
	await Promise.all(
		Array.from({ length: Math.min(size, items.length) }, async () => {
			while (i < items.length) await fn(items[i++]);
		})
	);
}

sw.addEventListener('install', (event) => {
	event.waitUntil(
		(async () => {
			const cache = await caches.open(PRE);
			const jobs = [
				...precacheUrls.code.map((u) => [u, 'default'] as const),
				...precacheUrls.statics.map((u) => [u, 'reload'] as const),
				...precacheUrls.shell.map((u) => [u, 'reload'] as const)
			];
			await pool(jobs, 8, async ([u, mode]) => {
				if (await cache.match(u)) return;
				const res = await fetch(u, { cache: mode });
				if (!res.ok) throw new Error(`預存失敗 ${res.status} ${u}`);
				await cache.put(u, await clean(res));
			});
		})()
	);
});

sw.addEventListener('activate', (event) => {
	event.waitUntil(
		(async () => {
			for (const name of await caches.keys()) {
				if (name.startsWith('precache-') && name !== PRE) await caches.delete(name);
			}
			await sw.clients.claim();
		})()
	);
});

const dataKey = (url: string) => {
	const u = new URL(url, sw.location.origin);
	u.search = '';
	return u.href;
};

function offlineMiss() {
	return new Response(JSON.stringify({ message: 'offline-miss' }), {
		status: 503,
		headers: { 'content-type': 'application/json' }
	});
}

async function handleNav(event: FetchEvent): Promise<Response> {
	const req = event.request;
	const cfg = await getCfg();
	const t0 = performance.now();
	const ctrl = new AbortController();
	let how = '網路';
	let res: Response | null = null;
	try {
		const net = (async () => {
			if (cfg.delay) await sleep(cfg.delay);
			return fetchNet(req, cfg, { signal: ctrl.signal });
		})();
		net.catch(() => {});
		let timer: ReturnType<typeof setTimeout>;
		const timeout = new Promise<null>((r) => (timer = setTimeout(() => r(null), cfg.navTimeout)));
		res = await Promise.race([net, timeout]);
		clearTimeout(timer!);
		if (res === null) {
			ctrl.abort();
			how = '空殼頁（逾時）';
		} else if (res.status >= 500) {
			how = '空殼頁（逾時）';
			res = null;
		}
	} catch {
		how = '空殼頁（離線）';
		res = null;
	}
	const ms = Math.round(performance.now() - t0);
	event.waitUntil(patchState({ nav: { how, ms, url: req.url, at: Date.now() } }));
	if (res) return res;
	const shell = (await (await caches.open(PRE)).match(SHELL)) ?? (await caches.match(SHELL));
	return shell ? clean(shell) : fetch(req);
}

async function handleData(event: FetchEvent): Promise<Response> {
	const req = event.request;
	const cfg = await getCfg();
	const cache = await caches.open(DATA);
	try {
		const res = await fetchNet(req, cfg);
		if (res.ok) event.waitUntil(cache.put(dataKey(req.url), res.clone()));
		return res;
	} catch {
		return (await cache.match(dataKey(req.url))) ?? offlineMiss();
	}
}

async function handleStatic(event: FetchEvent): Promise<Response> {
	const hit = await (await caches.open(PRE)).match(event.request, { ignoreSearch: true });
	if (hit) return hit;
	return fetchNet(event.request, await getCfg());
}

sw.addEventListener('fetch', (event) => {
	const req = event.request;
	const url = new URL(req.url);
	if (req.method !== 'GET' || url.origin !== sw.location.origin || !url.pathname.startsWith(BASE + '/')) return;
	if (req.mode === 'navigate') return event.respondWith(handleNav(event));
	if (url.pathname.endsWith('/__data.json')) return event.respondWith(handleData(event));
	event.respondWith(handleStatic(event));
});

// ---- 看過就存、當季預先下載 ----

async function saveData(url: string) {
	const cfg = await getCfg();
	const u = new URL(url, sw.location.origin);
	if (!u.pathname.endsWith('/')) u.pathname += '/';
	u.pathname += '__data.json';
	u.search = '';
	const res = await fetchNet(u.href, cfg, { cache: 'no-cache' });
	if (res.ok) await (await caches.open(DATA)).put(dataKey(u.href), res);
}

let predlRunning = false;

async function broadcast(msg: unknown) {
	for (const c of await sw.clients.matchAll()) c.postMessage(msg);
}

async function predownload(urls: string[]) {
	if (predlRunning) return;
	predlRunning = true;
	const cfg = await getCfg();
	const cache = await caches.open(DATA);
	const p: Predl = { version, done: 0, total: urls.length, failed: 0, running: true };
	await patchState({ predl: p });
	try {
		await pool(urls, 6, async (u) => {
			try {
				const res = await fetchNet(u, cfg, { cache: 'no-cache' });
				if (!res.ok) throw new Error(String(res.status));
				await cache.put(dataKey(u), res);
			} catch {
				p.failed++;
			}
			p.done++;
			if (p.done % 25 === 0) {
				await patchState({ predl: p });
				await broadcast({ type: 'predl', predl: p });
			}
		});
	} finally {
		p.running = false;
		predlRunning = false;
		await patchState({ predl: p });
		await broadcast({ type: 'predl', predl: p });
	}
}

async function info() {
	const state = await getState();
	return {
		version,
		nav: state.nav,
		predl: state.predl && { ...state.predl, running: predlRunning },
		cfg: await getCfg(),
		preCount: (await (await caches.open(PRE)).keys()).length,
		dataCount: (await (await caches.open(DATA)).keys()).length,
		precacheUrls: { code: precacheUrls.code.length, statics: precacheUrls.statics.length }
	};
}

sw.addEventListener('message', (event) => {
	const msg = event.data;
	if (msg?.type === 'skipWaiting') {
		event.waitUntil(sw.skipWaiting());
	} else if (msg?.type === 'info') {
		event.waitUntil(info().then((i) => event.ports[0]?.postMessage(i)));
	} else if (msg?.type === 'save-data') {
		event.waitUntil(saveData(msg.url).catch(() => {}));
	} else if (msg?.type === 'predownload') {
		event.waitUntil(predownload(msg.urls));
	}
});
