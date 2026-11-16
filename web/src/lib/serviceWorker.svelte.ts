import { dev } from '$app/env';
import { resolve } from '$app/paths';
import { countsAsReopen, seasonPagePaths, type WorkerMessage } from '#lib/offline.js';
import { currentSeason, matchEntries, playerEntries } from '#lib/site.js';

// off：dev、不支援 service worker 或註冊失敗；preparing：還沒被接手，或離線資料還沒備好；ready：備好了
export const connection = $state<{ offline: boolean; status: 'off' | 'preparing' | 'ready' }>({
	offline: false,
	status: 'off'
});

// 新版的檔案抓完、停在等待狀態，等使用者按更新條
export const update = $state({ ready: false });

// 建置時 VITE_SW_REMOVE=1 是自我移除版（docs/ops/service-worker.md）：頁面不註冊、不使用 service worker，否則會把它裝回去
const removing = import.meta.env.VITE_SW_REMOVE === '1';

const supported = () => !removing && !dev && 'serviceWorker' in navigator;

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

let registration: ServiceWorkerRegistration | undefined;
// 這一頁一開始就有 controller（或已被接手過）：之後 controller 再變就是換版，不是第一次安裝
let controlled = false;

export function applyUpdate(worker = registration?.waiting) {
	worker?.postMessage({ type: 'activate' } satisfies WorkerMessage);
}

// 離開前的時間；只放記憶體，頁面被系統回收的話重新開啟本來就是全新載入
let hiddenAt: number | undefined;
// 最近一輪檢查視同重新開啟：新版一抓完就直接換
let reopening = false;
let latestUpdateCheck = 0;

// 沒有 controller 時是第一次安裝，不算換版
function onInstalled(worker: ServiceWorker | null) {
	if (!navigator.serviceWorker.controller) return;
	if (reopening) return applyUpdate(worker);
	update.ready = true;
}

function watch(worker: ServiceWorker | null) {
	worker?.addEventListener('statechange', () => {
		if (worker.state === 'installed') onInstalled(worker);
	});
}

// registration.update() 不受 HTTP 快取影響（註冊時 updateViaCache: 'none'）；離線等失敗就算了。
// reopen 只對這一輪檢查有效：這一輪沒找到新版就收回，之後才出現的新版（例如別的分頁發的檢查找到的）照常只跳更新條
async function checkForUpdate(reopen = false) {
	const id = ++latestUpdateCheck;
	reopening = reopen;
	await registration?.update().catch(() => {});
	if (id === latestUpdateCheck && !registration?.installing) reopening = false;
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
		if (document.visibilityState !== 'visible') {
			hiddenAt = Date.now();
			return;
		}
		void checkConnection();
		if (!supported()) return;
		sync();
		const reopen = countsAsReopen(isStandalone(), hiddenAt === undefined ? 0 : Date.now() - hiddenAt);
		hiddenAt = undefined;
		if (reopen && update.ready) applyUpdate();
		void checkForUpdate(reopen);
	});
	// 這裡直接看常數，自我移除版的產物才不會留下註冊的程式（scripts/verify-sw-removal.sh 會檢查）
	if (removing || !supported()) return;
	connection.status = 'preparing';
	controlled = navigator.serviceWorker.controller !== null;
	navigator.serviceWorker.addEventListener('controllerchange', () => {
		if (controlled) return location.reload();
		controlled = true;
		onControlled();
	});
	const scope = resolve('/');
	navigator.serviceWorker.register(`${scope}service-worker.js`, { scope, type: 'module', updateViaCache: 'none' }).then(
		(reg) => {
			registration = reg;
			reg.addEventListener('updatefound', () => watch(reg.installing));
			watch(reg.installing);
			if (reg.waiting) onInstalled(reg.waiting);
			void checkForUpdate();
			sync();
		},
		() => {
			connection.status = 'off';
		}
	);
}
