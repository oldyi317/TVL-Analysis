// service worker、頁面、設定共用；不依賴 DOM，也不 import 任何 $app/*

// adapter-static fallback 產出的空殼頁檔名（避開 GitHub Pages 用的 404.html）
export const SHELL = 'shell.html';

// 資料檔離線又沒存時，service worker 回的錯誤訊息；根層 +error.svelte 認它
export const OFFLINE_MISS = 'offline-miss';

// 單場頁與球員頁的資料檔放這個快取：名稱不帶版本，換版不清，舊賽季看過的才留得住
export const DATA_CACHE = 'data';

// 預先下載的並行上限
export const PREDOWNLOAD_CONCURRENCY = 6;

// 頁面送給 service worker 的訊息：save 存一頁的資料檔；sync 確認預存還在，paths 不是 null 就預先下載（附 MessageChannel 的 port，回 { ready }）
export type WorkerMessage = { type: 'save'; path: string } | { type: 'sync'; paths: string[] | null };

export function dataUrlOf(pagePath: string): string {
	return `${pagePath.endsWith('/') ? pagePath : `${pagePath}/`}__data.json`;
}

// Kit 站內導覽時可能在資料檔網址加 ?x-sveltekit-invalidated=…，存與取都去掉
export function dataKey(url: string): string {
	return url.split(/[?#]/)[0];
}

type Entries = { season: string; group: string };

export function seasonPagePaths(
	scope: string,
	season: string,
	matchEntries: (Entries & { game: string })[],
	playerEntries: (Entries & { player: string })[]
): string[] {
	const prefix = (e: Entries) => `${scope}${encodeURIComponent(e.season)}/${e.group}`;
	return [
		...matchEntries.filter((e) => e.season === season).map((e) => `${prefix(e)}/matches/${e.game}/`),
		...playerEntries.filter((e) => e.season === season).map((e) => `${prefix(e)}/players/${e.player}/`)
	];
}

type PredownloadOptions = {
	store: Pick<Cache, 'match' | 'put'>;
	fetch: (url: string, init: RequestInit) => Promise<Response>;
	urls: string[];
	version: string;
	markerUrl: string;
};

// 記號是這一版的 version 就不抓；全部抓成功才寫記號，有失敗下次整輪再來
export async function predownload({ store, fetch, urls, version, markerUrl }: PredownloadOptions): Promise<boolean> {
	const marker = await store.match(markerUrl);
	if (marker && (await marker.json()).version === version) return true;
	const queue = [...urls];
	let failed = false;
	const worker = async () => {
		for (let url = queue.shift(); url !== undefined; url = queue.shift()) {
			try {
				const res = await fetch(url, { cache: 'no-cache' });
				if (res.ok) await store.put(dataKey(url), res);
				else failed = true;
			} catch {
				failed = true;
			}
		}
	};
	await Promise.all(Array.from({ length: PREDOWNLOAD_CONCURRENCY }, worker));
	if (failed) return false;
	// 不用 Response.json()：Safari 17 才有，iOS 16.4–16.x 會在這裡丟錯，記號永遠寫不進去
	await store.put(markerUrl, new Response(JSON.stringify({ version }), { headers: { 'content-type': 'application/json' } }));
	return true;
}

// 頁面不會載入的檔：預存了只是浪費；清單裡有一個檔抓不到整個安裝就失敗，所以 dotfile 也不放
const SKIPPED = new Set(['push/matches.json', 'og-image.png', 'og-image.svg', 'robots.txt']);

// 輸入 $app/manifest 的 assets 路徑（不含 base、不含開頭斜線），回傳要預存的；預設留下，新加的靜態檔離線時才不缺
export function precachedAssets(paths: string[]): string[] {
	return paths.filter(
		(path) => !path.startsWith('icons/') && !path.startsWith('.') && !SKIPPED.has(path)
	);
}
