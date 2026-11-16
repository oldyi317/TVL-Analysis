import { dev } from '$app/env';
import { resolve } from '$app/paths';
import { seasonPagePaths, type WorkerMessage } from '#lib/offline.js';
import { currentSeason, matchEntries, playerEntries } from '#lib/site.js';

// off：dev、不支援 service worker 或註冊失敗；preparing：還沒被接手，或離線資料還沒備好；ready：備好了
export const connection = $state<{ offline: boolean; status: 'off' | 'preparing' | 'ready' }>({
	offline: false,
	status: 'off'
});

const supported = () => !dev && 'serviceWorker' in navigator;

// 還沒被接手時看過的頁，接手後補存
const pending = new Set<string>();

// iOS 另有非標準的 navigator.standalone，兩個都看
export function isStandalone(): boolean {
	const ios = (navigator as Navigator & { standalone?: boolean }).standalone === true;
	return ios || matchMedia('(display-mode: standalone)').matches;
}

function send(message: WorkerMessage) {
	navigator.serviceWorker.controller?.postMessage(message);
}

function sync() {
	const controller = navigator.serviceWorker.controller;
	if (!controller) return;
	connection.status = 'preparing';
	const paths = isStandalone()
		? seasonPagePaths(resolve('/'), currentSeason.name, matchEntries(), playerEntries())
		: null;
	const { port1, port2 } = new MessageChannel();
	port1.onmessage = (event) => {
		connection.status = event.data?.ready ? 'ready' : 'preparing';
	};
	const message: WorkerMessage = { type: 'sync', paths };
	controller.postMessage(message, [port2]);
}

function onControlled() {
	for (const path of pending) send({ type: 'save', path });
	pending.clear();
	sync();
}

// 單場頁與球員頁的資料：整頁載入時內嵌在 HTML，要請 service worker 另外存；站內導覽時 service worker 抓資料檔就已經存了
export function rememberPage(path: string, navigationType: string) {
	if (!supported()) return;
	if (!navigator.serviceWorker.controller) {
		pending.add(path);
		return;
	}
	if (navigationType === 'enter') send({ type: 'save', path });
}

let latestCheck = 0;

// navigator.onLine 在 iOS 上不保證準，直接向自己的站台要 version.json；同時有好幾次探測時以最後發出的為準
export async function checkConnection() {
	const id = ++latestCheck;
	let offline = false;
	try {
		await fetch(`${resolve('/')}_app/version.json`, { cache: 'no-store', signal: AbortSignal.timeout(5000) });
	} catch {
		offline = true;
	}
	if (id === latestCheck) connection.offline = offline;
}

// 連線判斷不分環境都做；service worker 的範圍與路徑從 base 推導，dev 沒有建置出的 service worker 所以不註冊
export function startServiceWorker() {
	void checkConnection();
	window.addEventListener('online', () => void checkConnection());
	window.addEventListener('offline', () => {
		connection.offline = true;
		void checkConnection();
	});
	// iPhone 的獨立視窗從背景回來不會重新載入：回到前景就是下一次開啟，離線資料在背景時被清掉或上一輪沒抓完都在這時補
	document.addEventListener('visibilitychange', () => {
		if (document.visibilityState !== 'visible') return;
		void checkConnection();
		if (supported()) sync();
	});
	if (!supported()) return;
	connection.status = 'preparing';
	navigator.serviceWorker.addEventListener('controllerchange', onControlled);
	const scope = resolve('/');
	navigator.serviceWorker.register(`${scope}service-worker.js`, { scope, type: 'module' }).then(
		() => sync(),
		() => {
			connection.status = 'off';
		}
	);
}
