import { type Page } from '@playwright/test';
import { UA } from '../src/lib/installGuide.fixtures.js';
import { BASE, asStandalone, currentSeason as current, expect, openMoreSheet, test, topBar } from './helpers.js';

const URL_PATH = `${BASE}/notifications/`;
const PUSH = 'https://push.vstat.tw/**';
const PREFS_KEY = 'tvl.push.prefs';
const REGISTERED_KEY = 'tvl.push.registeredAt';

const OFFLINE_TEXT = '連上網路後才能更改';
const FAILED_TEXT = '目前無法開啟通知，請稍後再試';
const DENIED_TEXT = '你先前選了不允許通知。請到系統設定允許『VStat 企排』的通知，再回來開啟。';
const PRIVACY =
	'開啟後，本站會存下這台裝置的推播位址（由 Apple、Google 等推播服務產生）與你選的通知種類、組別，存放在 Cloudflare。不存姓名、帳號或任何能認出你的資料。兩種通知都關掉就會刪除；推播服務回報位址失效時也會刪除。';

type Permission = 'default' | 'granted' | 'denied';
type Fake = {
	permission?: Permission;
	// 使用者在系統詢問裡的選擇
	asked?: Permission;
	subscribed?: boolean;
	prefs?: { results: boolean; today: boolean; group: 'men' | 'women' | 'both' };
	noPush?: boolean;
};
type Call = { path: string; body: Record<string, unknown> };

const resultsSwitch = (page: Page) => page.getByRole('switch', { name: '賽果摘要' });
const todaySwitch = (page: Page) => page.getByRole('switch', { name: '今日賽事' });
const menBox = (page: Page) => page.getByRole('checkbox', { name: '男子組' });
const womenBox = (page: Page) => page.getByRole('checkbox', { name: '女子組' });
const status = (page: Page) => page.locator('.status');
const sheet = (page: Page) => page.getByRole('dialog', { name: '更多' });

// 瀏覽器端的呼叫記在 sessionStorage（重新整理後還在）；發送服務的請求在 Node 這一側攔截記錄
const browserLog = (page: Page): Promise<string[]> =>
	page.evaluate(() => JSON.parse(sessionStorage.getItem('fake.log') ?? '[]'));

// e2e 預設封鎖 service worker，所以 page.route 攔得到；service worker 的 ready 與 getRegistration 也一併換成假的
async function setup(page: Page, fake: Fake = {}) {
	const server = { status: 200, abort: false, calls: [] as Call[] };
	await page.route(PUSH, async (route) => {
		const request = route.request();
		const headers = { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'content-type' };
		if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers });
		if (server.abort) return route.abort('connectionfailed');
		server.calls.push({ path: new URL(request.url()).pathname, body: request.postDataJSON() });
		const json = server.status === 200 ? { ok: true } : { error: 'subscription_limit_reached' };
		await route.fulfill({ status: server.status, headers, json });
	});
	await page.addInitScript((o) => {
		/* eslint-disable @typescript-eslint/no-explicit-any */
		const w = window as any;
		if (o.noPush) delete w.PushManager;
		const session = sessionStorage;
		if (!session.getItem('fake.init')) {
			session.setItem('fake.init', '1');
			session.setItem('fake.permission', o.permission);
			if (o.subscribed) session.setItem('fake.sub', '1');
		}
		const log = (entry: string) => {
			const all = JSON.parse(session.getItem('fake.log') ?? '[]');
			all.push(entry);
			session.setItem('fake.log', JSON.stringify(all));
		};
		class FakeNotification {
			constructor() {
				log('new Notification');
			}
			static get permission() {
				return session.getItem('fake.permission');
			}
			static async requestPermission() {
				// 使用者啟動狀態會延續幾秒，延後呼叫也是 true；window.event 只在事件派送當下有值
				log(`requestPermission:${w.event?.type ?? 'outside-event'}`);
				session.setItem('fake.permission', o.asked);
				return o.asked;
			}
		}
		Object.defineProperty(w, 'Notification', { value: FakeNotification, configurable: true, writable: true });
		const endpoint = 'https://fcm.googleapis.com/fcm/send/fake';
		const subscription = {
			endpoint,
			toJSON: () => ({ endpoint, expirationTime: null, keys: { p256dh: 'p', auth: 'a' } }),
			unsubscribe: async () => {
				log('sub.unsubscribe');
				session.removeItem('fake.sub');
				return true;
			}
		};
		const registration = {
			pushManager: {
				getSubscription: async () => (session.getItem('fake.sub') ? subscription : null),
				subscribe: async (options: { userVisibleOnly: boolean; applicationServerKey: Uint8Array }) => {
					log(`subscribe:${options.userVisibleOnly}:${options.applicationServerKey.length}`);
					session.setItem('fake.sub', '1');
					return subscription;
				}
			}
		};
		Object.defineProperty(navigator.serviceWorker, 'ready', { value: Promise.resolve(registration) });
		(navigator.serviceWorker as any).getRegistration = async () => registration;
		try {
			if (o.prefs && !localStorage.getItem('tvl.push.prefs')) {
				localStorage.setItem('tvl.push.prefs', JSON.stringify(o.prefs));
			}
		} catch {}
	}, { permission: 'default', asked: 'granted', ...fake });
	return server;
}

const bothOn = { results: true, today: true, group: 'both' } as const;
const granted = { permission: 'granted', subscribed: true, prefs: bothOn } satisfies Fake;

test.describe('頁名與入口', () => {
	test('頁面標題與頂部列都是「通知」', async ({ page }) => {
		await setup(page);
		await page.goto(URL_PATH);
		await expect(page).toHaveTitle('通知｜VStat 企排');
		await expect(topBar(page).getByText('通知', { exact: true })).toBeVisible();
	});

	test('「更多」面板順序是球員、排行榜、歷史紀錄、通知、加到主畫面，點通知進到本頁', async ({ page }) => {
		await setup(page);
		await page.goto(`${BASE}/${current}/men/`);
		await openMoreSheet(page);
		await expect(sheet(page).getByRole('link')).toHaveText(['球員', '排行榜', '歷史紀錄', '通知', '加到主畫面']);
		await expect(sheet(page).getByRole('link', { name: '通知' })).toHaveAttribute('href', URL_PATH);
		await sheet(page).getByRole('link', { name: '通知' }).click();
		await expect(page).toHaveURL(URL_PATH);
	});

	test.describe('桌面', () => {
		test.use({ viewport: { width: 1440, height: 900 }, userAgent: UA.desktopChrome, isMobile: false, hasTouch: false });

		test('頁尾有「通知」連結，側欄沒有新項目', async ({ page }) => {
			await setup(page);
			await page.goto(`${BASE}/${current}/men/`);
			await expect(page.getByRole('contentinfo').getByRole('link', { name: '通知' })).toHaveAttribute('href', URL_PATH);
			const nav = page.getByRole('banner').getByRole('navigation', { name: '頁面' });
			await expect(nav.getByRole('link')).toHaveText(['首頁', '賽事', '戰績', '球隊', '球員', '排行榜', '歷史紀錄']);
		});

		test('桌面不顯示「加到主畫面」一節', async ({ page }) => {
			await setup(page);
			await page.goto(URL_PATH);
			await expect(resultsSwitch(page)).toBeVisible();
			await expect(page.getByRole('heading', { name: '加到主畫面' })).toHaveCount(0);
		});
	});
});

test.describe('瀏覽器不支援', () => {
	test('沒有開關，寫「加到主畫面後才能開啟通知」，下接安裝引導', async ({ page }) => {
		await setup(page, { noPush: true });
		await page.goto(URL_PATH);
		await expect(page.getByText('加到主畫面後才能開啟通知', { exact: true })).toBeVisible();
		await expect(page.getByText('請從 Chrome 的選單把本站加到主畫面')).toBeVisible();
		await expect(page.getByRole('switch')).toHaveCount(0);
		await expect(page.getByText('開啟後，本站會存下')).toHaveCount(0);
	});
});

test.describe('支援，還沒問過權限', () => {
	test('載入時不問；兩個開關都關；隱私說明在開關下方、逐字；Android 分頁多一節加到主畫面', async ({ page }) => {
		await setup(page);
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).not.toBeChecked();
		await expect(todaySwitch(page)).not.toBeChecked();
		await expect(page.getByText('有新賽果進站時通知', { exact: true })).toBeVisible();
		await expect(page.getByText('比賽日上午 10 點通知當天各場的開賽時間與賽前勝率', { exact: true })).toBeVisible();
		await expect(menBox(page)).toHaveCount(0);
		await expect(page.getByText(PRIVACY, { exact: true })).toBeVisible();
		const switchBottom = (await todaySwitch(page).boundingBox())!.y;
		const privacyTop = (await page.getByText(PRIVACY, { exact: true }).boundingBox())!.y;
		expect(privacyTop).toBeGreaterThan(switchBottom);
		await expect(page.getByRole('heading', { name: '加到主畫面' })).toBeVisible();
		await expect(page.getByText('請從 Chrome 的選單把本站加到主畫面')).toBeVisible();
		expect(await browserLog(page)).toEqual([]);
	});

	test('第一次打開才在點擊裡問；允許後登記，只顯示「已開啟」，沒有測試通知', async ({ page }) => {
		const server = await setup(page);
		await page.goto(URL_PATH);
		await resultsSwitch(page).click();
		await expect(resultsSwitch(page)).toBeChecked();
		await expect(status(page)).toHaveText('已開啟');
		await expect(todaySwitch(page)).not.toBeChecked();
		expect(await browserLog(page)).toEqual(['requestPermission:change', 'subscribe:true:65']);
		expect(server.calls).toEqual([
			{
				path: '/subscribe',
				body: {
					endpoint: 'https://fcm.googleapis.com/fcm/send/fake',
					expirationTime: null,
					keys: { p256dh: 'p', auth: 'a' },
					results: true,
					today: false,
					group: 'both'
				}
			}
		]);
	});

	test('系統詢問選「不允許」：開關彈回、停用，寫不允許的說明，不登記', async ({ page }) => {
		const server = await setup(page, { asked: 'denied' });
		await page.goto(URL_PATH);
		await todaySwitch(page).click();
		await expect(status(page)).toHaveText(DENIED_TEXT);
		await expect(todaySwitch(page)).not.toBeChecked();
		await expect(todaySwitch(page)).toBeDisabled();
		await expect(resultsSwitch(page)).toBeDisabled();
		expect(server.calls).toEqual([]);
	});

	test('關掉系統詢問（仍未決定）：開關彈回、不寫訊息、不登記', async ({ page }) => {
		const server = await setup(page, { asked: 'default' });
		await page.goto(URL_PATH);
		await resultsSwitch(page).click();
		await expect(resultsSwitch(page)).not.toBeChecked();
		await expect(resultsSwitch(page)).toBeEnabled();
		await expect(status(page)).toHaveText('');
		expect(server.calls).toEqual([]);
	});
});

test.describe('已允許', () => {
	test('開一個才出現組別，預設兩組都勾；改組別送出 group；只剩一組時那一格停用', async ({ page }) => {
		const server = await setup(page, { permission: 'granted' });
		await page.goto(URL_PATH);
		await expect(menBox(page)).toHaveCount(0);
		await expect(resultsSwitch(page)).not.toBeChecked();
		await resultsSwitch(page).click();
		await expect(menBox(page)).toBeChecked();
		await expect(womenBox(page)).toBeChecked();
		await expect(menBox(page)).toBeEnabled();
		expect(await browserLog(page)).toEqual(['subscribe:true:65']);
		await womenBox(page).uncheck();
		await expect(menBox(page)).toBeDisabled();
		await expect(womenBox(page)).toBeEnabled();
		expect(server.calls.map((c) => [c.path, c.body.group])).toEqual([
			['/subscribe', 'both'],
			['/subscribe', 'men']
		]);
		await womenBox(page).check();
		await expect(menBox(page)).toBeEnabled();
		expect(server.calls.at(-1)!.body.group).toBe('both');
	});

	test('只關一個：呼叫登記入口更新偏好，不退訂、不動權限', async ({ page }) => {
		const server = await setup(page, granted);
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).toBeChecked();
		await expect(todaySwitch(page)).toBeChecked();
		server.calls.length = 0;
		await resultsSwitch(page).uncheck();
		await expect(resultsSwitch(page)).not.toBeChecked();
		await expect(todaySwitch(page)).toBeChecked();
		expect(server.calls.map((c) => [c.path, c.body.results, c.body.today, c.body.group])).toEqual([
			['/subscribe', false, true, 'both']
		]);
		expect(await browserLog(page)).toEqual([]);
		await expect(menBox(page)).toBeVisible();
	});

	test('兩個都關：呼叫退訂入口、取消推播訂閱、組別收起', async ({ page }) => {
		const server = await setup(page, granted);
		await page.goto(URL_PATH);
		server.calls.length = 0;
		await resultsSwitch(page).uncheck();
		await expect(status(page)).toHaveText('');
		await todaySwitch(page).uncheck();
		await expect(menBox(page)).toHaveCount(0);
		expect(server.calls.map((c) => [c.path, c.body])).toEqual([
			['/subscribe', expect.objectContaining({ results: false, today: true })],
			['/unsubscribe', { endpoint: 'https://fcm.googleapis.com/fcm/send/fake' }]
		]);
		expect(await browserLog(page)).toEqual(['sub.unsubscribe']);
		await expect(resultsSwitch(page)).not.toBeChecked();
		await expect(todaySwitch(page)).not.toBeChecked();
	});

	test('推播訂閱被清掉而裝置上的偏好還是開：重開頁面，開關顯示關', async ({ page }) => {
		await setup(page, { ...granted, prefs: { results: true, today: false, group: 'both' } });
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).toBeChecked();
		await page.evaluate(() => sessionStorage.removeItem('fake.sub'));
		await page.reload();
		await expect(resultsSwitch(page)).not.toBeChecked();
		await expect(menBox(page)).toHaveCount(0);
		const prefs = await page.evaluate((key) => localStorage.getItem(key), PREFS_KEY);
		expect(JSON.parse(prefs!)).toMatchObject({ results: true });
	});

	test('推播訂閱被清掉後只打開今日賽事：不沿用裝置上記的賽果摘要', async ({ page }) => {
		const server = await setup(page, { permission: 'granted', prefs: { results: true, today: false, group: 'women' } });
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).not.toBeChecked();
		await todaySwitch(page).click();
		await expect(todaySwitch(page)).toBeChecked();
		await expect(resultsSwitch(page)).not.toBeChecked();
		expect(server.calls.map((c) => [c.path, c.body.results, c.body.today, c.body.group])).toEqual([
			['/subscribe', false, true, 'women']
		]);
	});

	test('有組別勾選時，隱私說明仍緊接在開關下方', async ({ page }) => {
		await setup(page, granted);
		await page.goto(URL_PATH);
		await expect(menBox(page)).toBeVisible();
		const switchBox = (await todaySwitch(page).boundingBox())!;
		const privacyTop = (await page.getByText(PRIVACY, { exact: true }).boundingBox())!.y;
		const groupsTop = (await menBox(page).boundingBox())!.y;
		expect(privacyTop).toBeGreaterThan(switchBox.y + switchBox.height);
		expect(groupsTop).toBeGreaterThan(privacyTop);
	});

	test('開站的重新登記還在途中時關掉兩個：等它送完才退訂', async ({ page }) => {
		await setup(page, { ...granted, prefs: { results: true, today: false, group: 'both' } });
		const events: string[] = [];
		await page.route(PUSH, async (route) => {
			if (route.request().method() === 'OPTIONS') return route.fallback();
			const path = new URL(route.request().url()).pathname;
			events.push(`${path}:start`);
			if (path === '/subscribe') await new Promise((resolve) => setTimeout(resolve, 2000));
			events.push(`${path}:end`);
			await route.fallback();
		});
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).toBeChecked();
		await expect.poll(() => events).toEqual(['/subscribe:start']);
		await resultsSwitch(page).uncheck();
		await expect(menBox(page)).toHaveCount(0);
		expect(events).toEqual(['/subscribe:start', '/subscribe:end', '/unsubscribe:start', '/unsubscribe:end']);
	});
});

test.describe('不允許、離線、登記失敗', () => {
	test('按過「不允許」：開關停用，寫逐字說明，沒有組別', async ({ page }) => {
		await setup(page, { permission: 'denied' });
		await page.goto(URL_PATH);
		await expect(status(page)).toHaveText(DENIED_TEXT);
		await expect(resultsSwitch(page)).toBeDisabled();
		await expect(todaySwitch(page)).toBeDisabled();
		await expect(menBox(page)).toHaveCount(0);
	});

	test('到系統設定改成允許再切回來：不用重新整理，開關就能用', async ({ page }) => {
		await setup(page, { permission: 'denied' });
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).toBeDisabled();
		await page.evaluate(() => {
			sessionStorage.setItem('fake.permission', 'granted');
			document.dispatchEvent(new Event('visibilitychange'));
		});
		await expect(resultsSwitch(page)).toBeEnabled();
		await expect(status(page)).toHaveText('');
	});

	test('離線：開關與組別停用，寫「連上網路後才能更改」', async ({ page, context }) => {
		await setup(page, granted);
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).toBeEnabled();
		await context.setOffline(true);
		await expect(status(page)).toHaveText(OFFLINE_TEXT);
		await expect(resultsSwitch(page)).toBeDisabled();
		await expect(todaySwitch(page)).toBeDisabled();
		await expect(menBox(page)).toBeDisabled();
		await expect(womenBox(page)).toBeDisabled();
	});

	test('登記失敗（筆數已滿 503）：開關彈回、寫「目前無法開啟通知」、新建的訂閱被取消、偏好不存', async ({ page }) => {
		const server = await setup(page, { permission: 'granted' });
		server.status = 503;
		await page.goto(URL_PATH);
		await resultsSwitch(page).click();
		await expect(status(page)).toHaveText(FAILED_TEXT);
		await expect(resultsSwitch(page)).not.toBeChecked();
		await expect(resultsSwitch(page)).toBeEnabled();
		await expect(menBox(page)).toHaveCount(0);
		expect(server.calls).toHaveLength(1);
		expect(await browserLog(page)).toEqual(['subscribe:true:65', 'sub.unsubscribe']);
		expect(await page.evaluate((key) => localStorage.getItem(key), PREFS_KEY)).toBeNull();
	});

	test('連不上發送服務但網站本身連得上：寫「目前無法開啟通知」', async ({ page }) => {
		const server = await setup(page, { permission: 'granted' });
		server.abort = true;
		await page.goto(URL_PATH);
		await resultsSwitch(page).click();
		await expect(status(page)).toHaveText(FAILED_TEXT);
		await expect(resultsSwitch(page)).not.toBeChecked();
	});

	test('連不上發送服務、網站也連不上：寫「連上網路後才能更改」', async ({ page }) => {
		const server = await setup(page, { permission: 'granted' });
		await page.goto(URL_PATH);
		server.abort = true;
		await page.route('**/_app/version.json', (route) => route.abort('connectionfailed'));
		await resultsSwitch(page).click();
		await expect(status(page)).toHaveText(OFFLINE_TEXT);
		await expect(resultsSwitch(page)).not.toBeChecked();
	});
});

test.describe('每日重新登記', () => {
	test('一天內開站多次只送一次；超過一天再送一次；用裝置上的偏好', async ({ page }) => {
		const server = await setup(page, { ...granted, prefs: { results: true, today: false, group: 'women' } });
		await page.goto(`${BASE}/${current}/men/`);
		await expect.poll(() => server.calls.length).toBe(1);
		expect(server.calls[0]).toMatchObject({
			path: '/subscribe',
			body: { endpoint: 'https://fcm.googleapis.com/fcm/send/fake', results: true, today: false, group: 'women' }
		});
		await page.reload();
		await page.reload();
		await page.waitForTimeout(300);
		expect(server.calls).toHaveLength(1);
		await page.evaluate((key) => localStorage.setItem(key, String(Date.now() - 25 * 3600_000)), REGISTERED_KEY);
		await page.reload();
		await expect.poll(() => server.calls.length).toBe(2);
		const recorded = await page.evaluate((key) => Number(localStorage.getItem(key)), REGISTERED_KEY);
		expect(Date.now() - recorded).toBeLessThan(60_000);
	});

	test('權限不是已允許時不重新登記', async ({ page }) => {
		const server = await setup(page, { ...granted, permission: 'default' });
		await page.goto(`${BASE}/${current}/men/`);
		await page.waitForTimeout(300);
		expect(server.calls).toEqual([]);
	});

	test('重新登記失敗不更新時間，下次開站再試', async ({ page }) => {
		const server = await setup(page, granted);
		server.status = 503;
		await page.goto(`${BASE}/${current}/men/`);
		await expect.poll(() => server.calls.length).toBe(1);
		expect(await page.evaluate((key) => localStorage.getItem(key), REGISTERED_KEY)).toBeNull();
		server.status = 200;
		await page.reload();
		await expect.poll(() => server.calls.length).toBe(2);
	});
});

test.describe('首頁提示卡', () => {
	const HOME = `${BASE}/${current}/men/`;
	const PROMPT_KEY = 'tvl.push.promptHidden';
	const card = (page: Page) => page.locator('.push-prompt');
	const settingsLink = (page: Page) => card(page).getByRole('link', { name: '設定通知' });
	const closeButton = (page: Page) => card(page).getByRole('button', { name: '關閉這則提示' });

	test('獨立視窗、支援、還沒問過權限、沒有推播訂閱：出現提示卡，文字逐字，連到通知頁', async ({ page }) => {
		await setup(page);
		await asStandalone(page);
		await page.goto(HOME);
		await expect(card(page)).toBeVisible();
		await expect(card(page).locator('p')).toHaveText('想在有新賽果、比賽日早上收到通知嗎？');
		await expect(settingsLink(page)).toHaveAttribute('href', URL_PATH);
		await expect(closeButton(page)).toBeVisible();
		for (const control of [settingsLink(page), closeButton(page)]) {
			const box = (await control.boundingBox())!;
			expect(box.height).toBeGreaterThanOrEqual(44);
			expect(box.width).toBeGreaterThanOrEqual(44);
		}
	});

	// 要不要顯示是問完瀏覽器才決定的：等首頁內容與兩個畫格，該出現的話這時已經出現
	async function expectNoCard(page: Page) {
		await expect(page.getByRole('region', { name: '戰績' })).toBeVisible();
		await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
		await expect(card(page)).toHaveCount(0);
	}

	for (const [name, fake, standalone] of [
		['在瀏覽器分頁內（Android Chrome）', {}, false],
		['瀏覽器不支援推播', { noPush: true }, true],
		['已經允許過通知', { permission: 'granted' }, true],
		['已經選過不允許', { permission: 'denied' }, true],
		['已有推播訂閱', { subscribed: true }, true]
	] as const) {
		test(`${name}：不出現提示卡`, async ({ page }) => {
			await setup(page, fake);
			if (standalone) await asStandalone(page);
			await page.goto(HOME);
			await expectNoCard(page);
		});
	}

	test('按關閉：提示卡消失並記在裝置上，重新載入不再出現', async ({ page }) => {
		await setup(page);
		await asStandalone(page);
		await page.goto(HOME);
		await closeButton(page).click();
		await expect(card(page)).toHaveCount(0);
		expect(await page.evaluate((key) => localStorage.getItem(key), PROMPT_KEY)).toBe('1');
		await page.reload();
		await expectNoCard(page);
	});

	test('從提示卡點進通知頁，沒開啟通知就回首頁：不再出現', async ({ page }) => {
		await setup(page);
		await asStandalone(page);
		await page.goto(HOME);
		await settingsLink(page).click();
		await expect(page).toHaveURL(URL_PATH);
		await expect(resultsSwitch(page)).not.toBeChecked();
		await page.goBack();
		await expect(page).toHaveURL(HOME);
		await expectNoCard(page);
		expect(await browserLog(page)).toEqual([]);
	});

	test('直接載入過通知頁：之後開首頁不出現', async ({ page }) => {
		await setup(page);
		await asStandalone(page);
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).not.toBeChecked();
		expect(await page.evaluate((key) => localStorage.getItem(key), PROMPT_KEY)).toBe('1');
		await page.goto(HOME);
		await expectNoCard(page);
	});

	// iPhone 實機（#285）：Safari 分頁留下的記號，加到主畫面後的 app 讀得到
	test('在瀏覽器分頁內進過通知頁：不記在裝置上，之後以獨立視窗開首頁仍出現提示卡', async ({ page, context }) => {
		await setup(page);
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).not.toBeChecked();
		expect(await page.evaluate((key) => localStorage.getItem(key), PROMPT_KEY)).toBeNull();
		const app = await context.newPage();
		await setup(app);
		await asStandalone(app);
		await app.goto(HOME);
		await expect(card(app)).toBeVisible();
	});

	test('localStorage 不可用：提示卡照樣出現，按關閉會消失，首頁與通知頁都不出錯', async ({ page }) => {
		await setup(page);
		await asStandalone(page);
		await page.addInitScript(() => {
			Object.defineProperty(window, 'localStorage', {
				get() {
					throw new DOMException('denied', 'SecurityError');
				}
			});
		});
		// e2e 封鎖 service worker 時 register() 回 undefined，站台註冊那一段每次載入都會丟這個錯，與 localStorage 無關
		const blockedWorker = "reading 'addEventListener'";
		const errors: string[] = [];
		page.on('pageerror', (error) => {
			if (!error.message.includes(blockedWorker)) errors.push(String(error));
		});
		await page.goto(HOME);
		await expect(card(page)).toBeVisible();
		await closeButton(page).click();
		await expect(card(page)).toHaveCount(0);
		await page.goto(URL_PATH);
		await expect(resultsSwitch(page)).toBeVisible();
		expect(errors).toEqual([]);
	});
});
