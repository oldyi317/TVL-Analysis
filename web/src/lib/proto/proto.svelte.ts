// 離線原型的頁面端邏輯：註冊 service worker、換版判斷、事件紀錄、除錯設定。拋棄式。
import { version as pageVersion } from '$app/env';
import { resolve } from '$app/paths';
import { matchEntries, playerEntries, currentSeason } from '#lib/site.js';
import site from '#lib/data/site.json';

export type Cfg = { navTimeout: number; delay: number; offline: boolean; threshold: number; standalone: boolean };
export const DEFAULT_CFG: Cfg = { navTimeout: 3000, delay: 0, offline: false, threshold: 30 * 60 * 1000, standalone: false };
type Nav = { how: string; ms: number; url: string; at: number };
type Predl = { version: string; done: number; total: number; failed: number; running: boolean };
export type LogEntry = { t: number; ev: string; away?: number; nav?: string; note?: string };

export const proto = $state({
	pageVersion,
	latest: '',
	swVersion: '',
	hasWaiting: false,
	controlled: false,
	online: true,
	cfg: { ...DEFAULT_CFG } as Cfg,
	nav: undefined as Nav | undefined,
	preCount: 0,
	dataCount: 0,
	predl: undefined as Predl | undefined,
	log: [] as LogEntry[],
	pending: false,
	status: ''
});

export const generatedMD = (() => {
	const m = /^\d+-(\d+)-(\d+)/.exec(site.generated_at);
	return m ? `${Number(m[1])}/${Number(m[2])}` : '';
})();

const BASE = resolve('/').replace(/\/$/, '');
const cfgKey = () => new Request(`${location.origin}${BASE}/__proto/config`);
const LOG_KEY = 'proto-log';
const HIDDEN_KEY = 'proto-hidden-at';

export const isStandalone = () =>
	proto.cfg.standalone ||
	matchMedia('(display-mode: standalone)').matches ||
	(navigator as Navigator & { standalone?: boolean }).standalone === true;
export const realStandalone = () =>
	matchMedia('(display-mode: standalone)').matches || (navigator as Navigator & { standalone?: boolean }).standalone === true;

function lsGet(key: string): string | null {
	try {
		return localStorage.getItem(key);
	} catch {
		return null;
	}
}
function lsSet(key: string, value: string | null) {
	try {
		if (value === null) localStorage.removeItem(key);
		else localStorage.setItem(key, value);
	} catch {
		/* ignore */
	}
}

function record(ev: string, extra: Partial<LogEntry> = {}) {
	const entry: LogEntry = { t: Date.now(), ev, nav: proto.nav ? `${proto.nav.how} ${proto.nav.ms}ms` : undefined, ...extra };
	const list: LogEntry[] = JSON.parse(lsGet(LOG_KEY) ?? '[]');
	list.push(entry);
	const kept = list.slice(-50);
	lsSet(LOG_KEY, JSON.stringify(kept));
	proto.log = kept.slice().reverse();
}

async function loadCfg() {
	try {
		const hit = await (await caches.open('proto-config')).match(cfgKey());
		if (hit) proto.cfg = { ...DEFAULT_CFG, ...(await hit.json()) };
	} catch {
		/* ignore */
	}
}

export async function setCfg(patch: Partial<Cfg>) {
	proto.cfg = { ...proto.cfg, ...patch };
	try {
		await (await caches.open('proto-config')).put(
			cfgKey(),
			new Response(JSON.stringify(proto.cfg), { headers: { 'content-type': 'application/json' } })
		);
	} catch {
		/* ignore */
	}
	if (patch.standalone) void maybePredownload();
}

function ask(worker: ServiceWorker | null | undefined, msg: unknown, timeout = 2500): Promise<any> {
	return new Promise((resolveFn) => {
		if (!worker) return resolveFn(undefined);
		const ch = new MessageChannel();
		const timer = setTimeout(() => resolveFn(undefined), timeout);
		ch.port1.onmessage = (e) => {
			clearTimeout(timer);
			resolveFn(e.data);
		};
		worker.postMessage(msg, [ch.port2]);
	});
}

const getReg = () => navigator.serviceWorker.getRegistration(`${BASE}/`);

export async function refreshInfo() {
	if (!('serviceWorker' in navigator)) return;
	const reg = await getReg();
	proto.controlled = !!navigator.serviceWorker.controller;
	proto.hasWaiting = !!reg?.waiting;
	const worker = navigator.serviceWorker.controller ?? reg?.active;
	const i = await ask(worker, { type: 'info' });
	if (!i) return;
	proto.swVersion = i.version;
	proto.nav = i.nav;
	proto.predl = i.predl;
	proto.preCount = i.preCount;
	proto.dataCount = i.dataCount;
}

function predlUrls(): string[] {
	const season = currentSeason.name;
	return [
		...matchEntries().filter((e) => e.season === season).map((e) => `${BASE}/${e.season}/${e.group}/matches/${e.game}/__data.json`),
		...playerEntries().filter((e) => e.season === season).map((e) => `${BASE}/${e.season}/${e.group}/players/${e.player}/__data.json`)
	];
}

export async function maybePredownload() {
	if (!isStandalone() || !navigator.serviceWorker?.controller) return;
	await refreshInfo();
	const p = proto.predl;
	if (p && p.version === proto.swVersion && !p.running && p.done >= p.total && p.total > 0) return;
	if (p?.running && p.version === proto.swVersion) return;
	navigator.serviceWorker.controller.postMessage({ type: 'predownload', urls: predlUrls() });
}

export function saveData(pathname: string) {
	navigator.serviceWorker?.controller?.postMessage({ type: 'save-data', url: pathname });
}

let wantReload = false;
let checking = false;

async function apply(reload: boolean) {
	wantReload = reload;
	proto.pending = false;
	const reg = await getReg();
	if (reg?.waiting) reg.waiting.postMessage({ type: 'skipWaiting' });
	else if (reload) location.reload();
}

export const applyUpdate = () => apply(true);

function waitForWaiting(reg: ServiceWorkerRegistration, ms: number): Promise<void> {
	return new Promise((done) => {
		if (reg.waiting) return done();
		const timer = setTimeout(done, ms);
		const watch = (w: ServiceWorker | null) => {
			if (!w) return;
			w.addEventListener('statechange', () => {
				if (w.state === 'installed' || w.state === 'redundant') {
					clearTimeout(timer);
					done();
				}
			});
		};
		watch(reg.installing);
		reg.addEventListener('updatefound', () => watch(reg.installing));
	});
}

async function checkUpdate(away: number) {
	if (!('serviceWorker' in navigator) || checking) return;
	checking = true;
	try {
		const res = await fetch(`${BASE}/_app/version.json`, { cache: 'no-store' });
		const latest = (await res.json()).version as string;
		proto.latest = latest;
		const reg = await getReg();
		if (!reg) return;
		await reg.update().catch(() => {});
		await refreshInfo();
		const swIsLatest = proto.swVersion === latest;
		const pageIsLatest = pageVersion === latest;
		if (swIsLatest && pageIsLatest) {
			proto.pending = false;
			proto.status = '已是最新';
			return;
		}
		if (!swIsLatest) {
			await waitForWaiting(reg, 30000);
			proto.hasWaiting = !!reg.waiting;
			if (!reg.waiting) return;
		}
		if (pageIsLatest) {
			proto.status = '頁面已是新版，新 service worker 直接接手';
			await apply(false);
		} else if (isStandalone() && away > proto.cfg.threshold) {
			proto.status = `離開 ${Math.round(away / 1000)} 秒，超過門檻，自動換新`;
			record('自動換新', { away });
			await apply(true);
		} else {
			proto.status = '有新版，等使用者點擊';
			proto.pending = true;
		}
	} catch (e) {
		proto.status = `檢查失敗：${String(e)}`;
	} finally {
		checking = false;
	}
}

export async function resetAll() {
	for (const name of await caches.keys()) await caches.delete(name);
	for (const reg of await navigator.serviceWorker.getRegistrations()) await reg.unregister();
	lsSet(LOG_KEY, null);
	lsSet(HIDDEN_KEY, null);
	location.reload();
}

let started = false;

export async function startProto() {
	if (started) return;
	started = true;
	proto.online = navigator.onLine;
	addEventListener('online', () => (proto.online = true));
	addEventListener('offline', () => (proto.online = false));
	proto.log = JSON.parse(lsGet(LOG_KEY) ?? '[]').reverse();
	await loadCfg();
	if (!('serviceWorker' in navigator)) {
		record('load', { note: '沒有 serviceWorker' });
		return;
	}

	navigator.serviceWorker.addEventListener('controllerchange', async () => {
		if (wantReload) return location.reload();
		await refreshInfo();
		void maybePredownload();
	});
	navigator.serviceWorker.addEventListener('message', (e) => {
		if (e.data?.type === 'predl') {
			proto.predl = e.data.predl;
			void refreshInfo();
		}
	});

	const hiddenAt = Number(lsGet(HIDDEN_KEY) ?? 0);
	const away = hiddenAt ? Date.now() - hiddenAt : 0;
	lsSet(HIDDEN_KEY, null);

	try {
		await navigator.serviceWorker.register(`${BASE}/service-worker.js`, { scope: `${BASE}/` });
	} catch (e) {
		proto.status = `註冊失敗：${String(e)}`;
	}
	await refreshInfo();
	record('頁面載入', { away: away || undefined, note: proto.controlled ? undefined : '尚無 SW 接手' });
	void maybePredownload();
	void checkUpdate(away);

	document.addEventListener('visibilitychange', () => {
		if (document.visibilityState === 'hidden') {
			lsSet(HIDDEN_KEY, String(Date.now()));
			record('hidden');
			return;
		}
		const at = Number(lsGet(HIDDEN_KEY) ?? 0);
		const gone = at ? Date.now() - at : 0;
		lsSet(HIDDEN_KEY, null);
		record('visible', { away: gone });
		void refreshInfo();
		void checkUpdate(gone);
	});
}
