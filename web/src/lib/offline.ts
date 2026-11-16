// service worker、頁面、設定共用；不依賴 DOM，也不 import 任何 $app/*

// adapter-static fallback 產出的空殼頁檔名（避開 GitHub Pages 用的 404.html）
export const SHELL = 'shell.html';

// 資料檔離線又沒存時，service worker 回的錯誤訊息；根層 +error.svelte 認它
export const OFFLINE_MISS = 'offline-miss';

// 頁面不會載入的檔：預存了只是浪費；清單裡有一個檔抓不到整個安裝就失敗，所以 dotfile 也不放
const SKIPPED = new Set(['push/matches.json', 'og-image.png', 'og-image.svg', 'robots.txt']);

// 輸入 $app/manifest 的 assets 路徑（不含 base、不含開頭斜線），回傳要預存的；預設留下，新加的靜態檔離線時才不缺
export function precachedAssets(paths: string[]): string[] {
	return paths.filter(
		(path) => !path.startsWith('icons/') && !path.startsWith('.') && !SKIPPED.has(path)
	);
}
