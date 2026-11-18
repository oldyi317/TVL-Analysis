import { browserStorage } from '#lib/groupMemory.js';
import {
	DEFAULT_PREFS,
	checksOf,
	groupOf,
	isDue,
	readPrefs,
	readRegisteredAt,
	registerBody,
	urlBase64ToBytes,
	writePrefs,
	writeRegisteredAt,
	type Prefs
} from '#lib/pushPrefs.js';
import { PUSH_PUBLIC_KEY, PUSH_URL } from '#lib/pushService.js';
import { checkConnection, connection } from '#lib/serviceWorker.svelte.js';

export const OFFLINE_TEXT = '連上網路後才能更改';
export const FAILED_TEXT = '目前無法開啟通知，請稍後再試';
export const DENIED_TEXT = '你先前選了不允許通知。請到系統設定允許『VStat 企排』的通知，再回來開啟。';

const TIMEOUT_MS = 10_000;

type Kind = 'results' | 'today';

// support 在瀏覽器裡判斷前是 null（prerender 的 HTML 不知道瀏覽器能力）
export const push = $state<{
	support: boolean | null;
	permission: NotificationPermission;
	subscribed: boolean;
	prefs: Prefs;
	busy: boolean;
	message: string;
}>({ support: null, permission: 'default', subscribed: false, prefs: { ...DEFAULT_PREFS }, busy: false, message: '' });

// 自我移除版沒有 service worker 可訂閱，算不支援
const removing = import.meta.env.VITE_SW_REMOVE === '1';

export function supportsPush(): boolean {
	return !removing && 'Notification' in window && 'PushManager' in window && 'serviceWorker' in navigator;
}

export function isOffline(): boolean {
	return connection.offline || (typeof navigator !== 'undefined' && !navigator.onLine);
}

// 開關顯示開，要瀏覽器真的有推播訂閱、裝置上的偏好也是開
export const isOn = (kind: Kind) => push.subscribed && push.prefs[kind];

export const groupChecks = () => checksOf(push.prefs.group);

// 沒有 service worker 註冊（dev、註冊失敗）時 ready 永遠不會好，逾時就算登記失敗
function readyRegistration(): Promise<ServiceWorkerRegistration> {
	return Promise.race([
		navigator.serviceWorker.ready,
		new Promise<never>((_, reject) => setTimeout(() => reject(new Error('service worker not ready')), TIMEOUT_MS))
	]);
}

async function post(path: string, body: object) {
	let response: Response;
	try {
		response = await fetch(`${PUSH_URL}${path}`, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify(body),
			signal: AbortSignal.timeout(TIMEOUT_MS)
		});
	} catch (error) {
		// navigator.onLine 在 iOS 上不保證準：連不上時向自己的站台探測，離線的話頁面改顯示離線說明
		if ((error as Error).name !== 'TimeoutError') await checkConnection();
		throw error;
	}
	if (!response.ok) throw new Error(`status ${response.status}`);
}

const currentSubscription = async () =>
	(await navigator.serviceWorker.getRegistration())?.pushManager.getSubscription() ?? null;

// 權限與有沒有推播訂閱每次都問瀏覽器；偏好才是裝置上記的
// 問完有沒有推播訂閱才顯示開關，免得已訂閱的人先看到關、還點得到
export async function refreshPush() {
	if (push.busy) return;
	if (!supportsPush()) {
		push.support = false;
		return;
	}
	push.permission = Notification.permission;
	push.prefs = readPrefs(browserStorage());
	try {
		push.subscribed = (await currentSubscription()) !== null;
	} catch {
		push.subscribed = false;
	}
	push.support = true;
}

async function register(next: Prefs) {
	const { pushManager } = await readyRegistration();
	let subscription = await pushManager.getSubscription();
	const created = !subscription;
	subscription ??= await pushManager.subscribe({
		userVisibleOnly: true,
		applicationServerKey: urlBase64ToBytes(PUSH_PUBLIC_KEY)
	});
	try {
		await post('/subscribe', registerBody(subscription.toJSON(), next));
	} catch (error) {
		if (created) await subscription.unsubscribe().catch(() => {});
		throw error;
	}
	push.subscribed = true;
	writeRegisteredAt(browserStorage(), Date.now());
}

async function unregister() {
	const subscription = await currentSubscription();
	if (subscription) {
		await post('/unsubscribe', { endpoint: subscription.endpoint });
		await subscription.unsubscribe().catch(() => {});
	}
	push.subscribed = false;
}

async function apply(next: Prefs, asking: Promise<NotificationPermission> | null, announce: boolean) {
	push.busy = true;
	push.message = '';
	try {
		await reregistering;
		push.permission = asking ? await asking : Notification.permission;
		if (next.results || next.today) {
			if (push.permission !== 'granted') return;
			await register(next);
		} else {
			await unregister();
		}
		push.prefs = next;
		writePrefs(browserStorage(), next);
		if (announce) push.message = '已開啟';
	} catch (error) {
		push.message = FAILED_TEXT;
	} finally {
		push.busy = false;
	}
}

// 系統詢問必須在點擊處理函式裡、第一個 await 之前發出，所以這裡先同步呼叫再交給 apply 等結果
export function setKind(kind: Kind, on: boolean): Promise<void> {
	if (push.busy) return Promise.resolve();
	const asking = on && Notification.permission === 'default' ? Notification.requestPermission() : null;
	// 沒有推播訂閱時畫面上兩個都是關，新偏好從關開始算，不沿用裝置上記的
	const shown = push.subscribed ? push.prefs : { ...push.prefs, results: false, today: false };
	return apply({ ...shown, [kind]: on }, asking, on);
}

export function setGroup(men: boolean, women: boolean): Promise<void> {
	if (push.busy) return Promise.resolve();
	return apply({ ...push.prefs, group: groupOf(men, women) }, null, false);
}

// 開站時的重新登記還在途中時，使用者的操作等它送完，舊偏好才不會後到、蓋掉新的
let reregistering: Promise<void> = Promise.resolve();

export function reregisterDaily(): Promise<void> {
	reregistering = reregisterIfDue();
	return reregistering;
}

// 有推播訂閱的裝置一天最多重新登記一次；失敗就留到下次開站再試
async function reregisterIfDue() {
	if (!supportsPush() || Notification.permission !== 'granted') return;
	const storage = browserStorage();
	const prefs = readPrefs(storage);
	if ((!prefs.results && !prefs.today) || !isDue(readRegisteredAt(storage), Date.now())) return;
	try {
		const subscription = await currentSubscription();
		if (!subscription) return;
		await post('/subscribe', registerBody(subscription.toJSON(), prefs));
		writeRegisteredAt(storage, Date.now());
	} catch {}
}
