// 發送服務送的是 Declarative Web Push 的 JSON（形狀見 push/README.md「通知內容」）。
// 系統自己會處理這種格式時 service worker 收不到事件；收得到就由這裡顯示同樣內容。
type Notice = { title: string; body: string; navigate: string };

type Registration = Pick<ServiceWorkerRegistration, 'showNotification' | 'scope'>;

const filled = (value: unknown): value is string => typeof value === 'string' && value !== '';

function isHttpUrl(value: unknown): value is string {
	if (typeof value !== 'string') return false;
	try {
		return /^https?:$/.test(new URL(value).protocol);
	} catch {
		return false;
	}
}

function noticeOf(data: { json(): unknown } | null): Notice | null {
	let payload: unknown;
	try {
		payload = data?.json();
	} catch {
		return null;
	}
	const notification = (payload as { notification?: Partial<Record<keyof Notice, unknown>> } | null)?.notification;
	if (typeof notification !== 'object' || notification === null) return null;
	const { title, body, navigate } = notification;
	if (!filled(title) || !filled(body) || !isHttpUrl(navigate)) return null;
	return { title, body, navigate };
}

// 內容不認得時仍要顯示一則：收到推播卻不顯示通知，Safari 會撤掉推播訂閱，Chrome 會自己補一則英文通知
// 不設 tag：同一天的賽果摘要與今日賽事要並存，不互相取代
export async function showPush(registration: Registration, data: { json(): unknown } | null): Promise<void> {
	const { title, body, navigate } = noticeOf(data) ?? {
		title: 'VStat 企排',
		body: '有新消息，點開查看',
		navigate: registration.scope
	};
	await registration.showNotification(title, { body, data: { navigate } });
}

type WindowLike = { focus(): Promise<WindowLike>; navigate(url: string): Promise<unknown> };
type ClientsLike = {
	matchAll(options: { type: 'window' }): Promise<readonly WindowLike[]>;
	openWindow(url: string): Promise<unknown>;
};

// 點通知：已有這個網站的視窗就聚焦它並導過去，沒有或導不過去就開新視窗。網址照通知帶的，不改寫
export async function openPush(clients: ClientsLike, data: unknown, scope: string): Promise<void> {
	const navigate = (data as { navigate?: unknown } | null)?.navigate;
	const url = typeof navigate === 'string' ? navigate : scope;
	// matchAll 照最近聚焦的順序排
	const [client] = await clients.matchAll({ type: 'window' });
	if (client) {
		try {
			// 聚焦要趁點擊的授權還在，所以先做
			await (await client.focus()).navigate(url);
			return;
		} catch {}
	}
	await clients.openWindow(url);
}
