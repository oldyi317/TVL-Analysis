import { describe, expect, it } from 'vitest';
import {
	AWAY_LIMIT_MS,
	PREDOWNLOAD_CONCURRENCY,
	countsAsReopen,
	dataKey,
	dataUrlOf,
	precachedAssets,
	predownload,
	seasonPagePaths,
	uninstall
} from './offline.js';

const MARKER = '/__offline/predownload.json';

function fakeStore() {
	const map = new Map<string, Response>();
	const keyOf = (r: RequestInfo | URL) => (typeof r === 'string' ? r : r instanceof URL ? r.href : r.url);
	return {
		map,
		match: async (r: RequestInfo | URL) => map.get(keyOf(r))?.clone(),
		put: async (r: RequestInfo | URL, res: Response) => void map.set(keyOf(r), res)
	};
}

function fakeFetch(opts: { delay?: number; fail?: (url: string) => boolean } = {}) {
	const calls: { url: string; cache: RequestCache | undefined }[] = [];
	let inFlight = 0;
	let maxInFlight = 0;
	const fn = async (input: RequestInfo | URL, init?: RequestInit) => {
		const url = String(input);
		calls.push({ url, cache: init?.cache });
		inFlight++;
		maxInFlight = Math.max(maxInFlight, inFlight);
		await new Promise((r) => setTimeout(r, opts.delay ?? 0));
		inFlight--;
		return opts.fail?.(url) ? new Response('x', { status: 500 }) : new Response(`{"u":"${url}"}`);
	};
	return { fn, calls, max: () => maxInFlight };
}

const urls = (n: number) => Array.from({ length: n }, (_, i) => `/2026-27/men/matches/${i}/__data.json`);

describe('dataUrlOf / dataKey', () => {
	it('頁面路徑換成資料檔網址，沒結尾斜線也補', () => {
		expect(dataUrlOf('/2026-27/men/matches/5/')).toBe('/2026-27/men/matches/5/__data.json');
		expect(dataUrlOf('/2026-27/men/matches/5')).toBe('/2026-27/men/matches/5/__data.json');
		expect(dataUrlOf('/tvl/2026-27/men/players/7/')).toBe('/tvl/2026-27/men/players/7/__data.json');
	});

	it('dataKey 去掉 search 與 hash', () => {
		expect(dataKey('/a/__data.json?x-sveltekit-invalidated=01#h')).toBe('/a/__data.json');
		expect(dataKey('http://localhost/a/__data.json?x=1')).toBe('http://localhost/a/__data.json');
		expect(dataKey('/a/__data.json')).toBe('/a/__data.json');
	});
});

describe('seasonPagePaths', () => {
	const matches = [
		{ season: '2026-27', group: 'men', game: '5' },
		{ season: '2025-26', group: 'men', game: '6' }
	];
	const players = [
		{ season: '2026-27', group: 'women', player: '9' },
		{ season: '2025-26', group: 'men', player: '8' }
	];

	it('只回該賽季的單場頁與球員頁', () => {
		expect(seasonPagePaths('/', '2026-27', matches, players)).toEqual([
			'/2026-27/men/matches/5/',
			'/2026-27/women/players/9/'
		]);
	});

	it('scope 帶 base、賽季編碼；換賽季後只回新賽季', () => {
		expect(seasonPagePaths('/tvl-shell-proto/', '2025-26', matches, players)).toEqual([
			'/tvl-shell-proto/2025-26/men/matches/6/',
			'/tvl-shell-proto/2025-26/men/players/8/'
		]);
		expect(seasonPagePaths('/', 'a b', [{ season: 'a b', group: 'men', game: '1' }], [])).toEqual([
			'/a%20b/men/matches/1/'
		]);
	});
});

describe('predownload', () => {
	it('首輪全抓，每個請求都用 no-cache，存進 store 並寫記號', async () => {
		const store = fakeStore();
		const f = fakeFetch();
		const ok = await predownload({ store, fetch: f.fn, urls: urls(3), version: 'v1', markerUrl: MARKER });
		expect(ok).toBe(true);
		expect(f.calls).toHaveLength(3);
		expect(f.calls.every((c) => c.cache === 'no-cache')).toBe(true);
		expect([...store.map.keys()].sort()).toEqual([MARKER, ...urls(3)].sort());
		expect(await (await store.match(MARKER))!.json()).toEqual({ version: 'v1' });
	});

	it('同 version 第二輪 0 個請求', async () => {
		const store = fakeStore();
		await predownload({ store, fetch: fakeFetch().fn, urls: urls(3), version: 'v1', markerUrl: MARKER });
		const f = fakeFetch();
		const ok = await predownload({ store, fetch: f.fn, urls: urls(3), version: 'v1', markerUrl: MARKER });
		expect(ok).toBe(true);
		expect(f.calls).toHaveLength(0);
	});

	it('version 改變後同一批 url 再請求一輪', async () => {
		const store = fakeStore();
		await predownload({ store, fetch: fakeFetch().fn, urls: urls(3), version: 'v1', markerUrl: MARKER });
		const f = fakeFetch();
		const ok = await predownload({ store, fetch: f.fn, urls: urls(3), version: 'v2', markerUrl: MARKER });
		expect(ok).toBe(true);
		expect(f.calls.map((c) => c.url).sort()).toEqual(urls(3));
		expect(await (await store.match(MARKER))!.json()).toEqual({ version: 'v2' });
	});

	it('並行數上限恰為 PREDOWNLOAD_CONCURRENCY（6）', async () => {
		expect(PREDOWNLOAD_CONCURRENCY).toBe(6);
		const f = fakeFetch({ delay: 5 });
		await predownload({ store: fakeStore(), fetch: f.fn, urls: urls(20), version: 'v1', markerUrl: MARKER });
		expect(f.calls).toHaveLength(20);
		expect(f.max()).toBe(6);
	});

	it('有一個失敗就不寫記號，下一輪整輪重抓', async () => {
		const store = fakeStore();
		const bad = urls(4)[2];
		const first = fakeFetch({ fail: (u) => u === bad });
		expect(await predownload({ store, fetch: first.fn, urls: urls(4), version: 'v1', markerUrl: MARKER })).toBe(false);
		expect(store.map.has(MARKER)).toBe(false);
		expect(store.map.has(bad)).toBe(false);
		const second = fakeFetch();
		expect(await predownload({ store, fetch: second.fn, urls: urls(4), version: 'v1', markerUrl: MARKER })).toBe(true);
		expect(second.calls).toHaveLength(4);
	});

	it('換季：對存著舊賽季資料檔的 store 預先下載新賽季，舊賽季條目仍在', async () => {
		const store = fakeStore();
		const oldUrl = dataUrlOf('/2025-26/men/matches/1/');
		await store.put(oldUrl, new Response('old'));
		const paths = seasonPagePaths('/', '2026-27', [{ season: '2026-27', group: 'men', game: '2' }], []);
		const f = fakeFetch();
		await predownload({ store, fetch: f.fn, urls: paths.map(dataUrlOf), version: 'v1', markerUrl: MARKER });
		expect(f.calls.map((c) => c.url)).toEqual(['/2026-27/men/matches/2/__data.json']);
		expect(await (await store.match(oldUrl))!.text()).toBe('old');
	});
});

describe('precachedAssets', () => {
	it('排除四張圖示、推播賽事檔與頁面不會載入的檔，其餘都留', () => {
		const input = [
			'favicon.png',
			'og-image.png',
			'og-image.svg',
			'robots.txt',
			'.nojekyll',
			'icons/apple-touch-icon.png',
			'icons/icon-192.png',
			'icons/icon-512.png',
			'icons/icon-maskable-512.png',
			'push/matches.json',
			'logos/men-1.png',
			'logos/icons-team.png',
			'myicons/a.png',
			'push/other.json',
			'data/push/matches.json'
		];
		expect(precachedAssets(input)).toEqual([
			'favicon.png',
			'logos/men-1.png',
			'logos/icons-team.png',
			'myicons/a.png',
			'push/other.json',
			'data/push/matches.json'
		]);
	});

	it('空清單回空清單', () => {
		expect(precachedAssets([])).toEqual([]);
	});
});

describe('countsAsReopen', () => {
	it('獨立視窗離開剛好 30 分鐘不算重新開啟，多 1 毫秒才算', () => {
		expect(AWAY_LIMIT_MS).toBe(30 * 60 * 1000);
		expect(countsAsReopen(true, 1_800_000)).toBe(false);
		expect(countsAsReopen(true, 1_800_001)).toBe(true);
	});

	it('瀏覽器分頁離開再久也不算', () => {
		expect(countsAsReopen(false, 24 * 60 * 60 * 1000)).toBe(false);
	});
});

describe('uninstall', () => {
	function fakeCaches(names: string[], opts: { failKeys?: boolean; failDelete?: string } = {}) {
		const log: string[] = [];
		const store = {
			keys: async () => {
				if (opts.failKeys) throw new Error('keys');
				return names;
			},
			delete: async (name: string) => {
				if (name === opts.failDelete) throw new Error('delete');
				log.push(`delete ${name}`);
				return true;
			}
		};
		const registration = {
			unregister: async () => {
				log.push('unregister');
				return true;
			}
		};
		return { store, registration, log };
	}

	it('刪掉全部 cache（不挑名稱），全部刪完才註銷', async () => {
		const f = fakeCaches(['precache-v1', 'offline-data', 'other']);
		await uninstall(f.store, f.registration);
		expect(f.log.slice(0, 3).sort()).toEqual(['delete offline-data', 'delete other', 'delete precache-v1']);
		expect(f.log[3]).toBe('unregister');
		expect(f.log).toHaveLength(4);
	});

	it('某個 cache 刪除失敗：其餘照刪，照樣註銷', async () => {
		const f = fakeCaches(['a', 'b', 'c'], { failDelete: 'b' });
		await uninstall(f.store, f.registration);
		expect(f.log.slice(0, 2).sort()).toEqual(['delete a', 'delete c']);
		expect(f.log[2]).toBe('unregister');
	});

	it('列不出 cache 時也照樣註銷', async () => {
		const f = fakeCaches([], { failKeys: true });
		await uninstall(f.store, f.registration);
		expect(f.log).toEqual(['unregister']);
	});
});
