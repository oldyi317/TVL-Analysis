import { type BrowserContext, type Page, type Worker } from '@playwright/test';
import { BASE, currentSeason, expect, loadAndWaitControlled, test } from './helpers.js';

test.use({ serviceWorkers: 'allow' });

const HOME = `${BASE}/${currentSeason}/men/`;

type Shown = { title: string; options: NotificationOptions | undefined };
type Recording = { shown: Shown[] };

// headless Chromium 的通知權限固定是 denied，真的 showNotification 會被拒：只把它換成記錄用的，其餘是建置出來的 service worker
async function recordNotifications(context: BrowserContext): Promise<Worker> {
	const [worker] = context.serviceWorkers();
	await worker.evaluate(() => {
		const scope = globalThis as unknown as Recording & { registration: ServiceWorkerRegistration };
		scope.shown = [];
		scope.registration.showNotification = async (title, options) => void scope.shown.push({ title, options });
	});
	return worker;
}

const shown = (worker: Worker) => worker.evaluate(() => (globalThis as unknown as Recording).shown);

// 用瀏覽器的偵錯介面把推播內容直接交給已註冊的 service worker，不經過推播服務
async function deliverPush(context: BrowserContext, page: Page, data: string) {
	const cdp = await context.newCDPSession(page);
	const registered = new Promise<string>((resolve) => {
		cdp.on('ServiceWorker.workerRegistrationUpdated', ({ registrations }) => {
			if (registrations[0]) resolve(registrations[0].registrationId);
		});
	});
	await cdp.send('ServiceWorker.enable');
	await cdp.send('ServiceWorker.deliverPushMessage', {
		origin: new URL(page.url()).origin,
		registrationId: await registered,
		data
	});
}

test('service worker 收到帶 notification 的推播，顯示標題與內文相同的通知，記下要開的網址', async ({ page, context }) => {
	await loadAndWaitControlled(page, HOME);
	const worker = await recordNotifications(context);
	const navigate = new URL(`${BASE}/${currentSeason}/men/matches/#d-2026-11-21`, page.url()).href;
	await deliverPush(
		context,
		page,
		JSON.stringify({ web_push: 8030, notification: { title: '男子組賽果', body: '台電 3:1 屏東台電', navigate } })
	);
	await expect
		.poll(() => shown(worker))
		.toEqual([{ title: '男子組賽果', options: { body: '台電 3:1 屏東台電', data: { navigate } } }]);
});

test('推播內容不是認得的格式：service worker 不丟錯，顯示一則開到站台首頁的泛用通知', async ({ page, context }) => {
	await loadAndWaitControlled(page, HOME);
	const worker = await recordNotifications(context);
	const errors: string[] = [];
	const cdp = await context.newCDPSession(page);
	cdp.on('ServiceWorker.workerErrorReported', ({ errorMessage }) => errors.push(errorMessage.errorMessage));
	await deliverPush(context, page, '不是 JSON');
	const root = new URL(`${BASE}/`, page.url()).href;
	await expect
		.poll(() => shown(worker))
		.toEqual([{ title: 'VStat 企排', options: { body: '有新消息，點開查看', data: { navigate: root } } }]);
	expect(errors).toEqual([]);
});
