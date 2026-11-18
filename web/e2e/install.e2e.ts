import { type Page } from '@playwright/test';
import { UA } from '../src/lib/installGuide.fixtures.js';
import { BASE, asStandalone, currentSeason as current, openMoreSheet, seasonButton, topBar, groupLink, expect, test } from './helpers.js';

const URL_PATH = `${BASE}/notifications/`;
const NAME = '通知';
const INSTALL_ROW = '加到主畫面';
const INSTALL = '安裝 VStat 企排';

// 這兩種引導只在瀏覽器不支援推播時出現；e2e 的 Chromium 支援，所以拿掉 PushManager
const withoutPush = (page: Page) => page.addInitScript(() => void delete (window as { PushManager?: unknown }).PushManager);

const installButton = (page: Page) => page.getByRole('button', { name: INSTALL });
const steps = (page: Page) => page.locator('main ol > li');
const sheet = (page: Page) => page.getByRole('dialog', { name: '更多' });
const moreButton = (page: Page) =>
	page.getByRole('navigation', { name: '主要分頁' }).getByRole('button', { name: '更多' });

type Probe = { __prompts?: number; __finish?: (() => void)[] };

// 在頁面裡送一個合成的 beforeinstallprompt；headless 叫不出原生對話框，prompt() 有沒有被呼叫就是接縫
async function fireInstallPrompt(page: Page) {
	return page.evaluate(() => {
		const w = window as unknown as Probe;
		w.__prompts ??= 0;
		const finish = (w.__finish ??= []);
		const choice = new Promise<{ outcome: string }>((done) => {
			finish.push(() => done({ outcome: 'dismissed' }));
		});
		const event = Object.assign(new Event('beforeinstallprompt', { cancelable: true }), {
			prompt: () => {
				w.__prompts!++;
				return Promise.resolve();
			},
			userChoice: choice
		});
		window.dispatchEvent(event);
		return event.defaultPrevented;
	});
}

// 第 n 個事件的原生對話框被使用者關掉
const finishChoice = (page: Page, n: number) => page.evaluate((i) => (window as unknown as Probe).__finish![i](), n);
const promptCalls = (page: Page) => page.evaluate(() => (window as unknown as Probe).__prompts);

test.describe('頂部列與分頁列', () => {
	test('頂部列只有返回鈕與頁名；分頁列「更多」是目前所在', async ({ page }) => {
		await page.goto(URL_PATH);
		await expect(topBar(page).getByRole('link', { name: '返回' })).toBeVisible();
		await expect(topBar(page).getByText(NAME, { exact: true })).toBeVisible();
		await expect(topBar(page).getByRole('link', { name: 'VStat 企排' })).toHaveCount(0);
		await expect(groupLink(page, '男子')).toHaveCount(0);
		await expect(groupLink(page, '女子')).toHaveCount(0);
		await expect(seasonButton(page)).toHaveCount(0);
		await expect(moreButton(page)).toHaveAttribute('aria-current', 'page');
	});

	test('從列表頁開「更多」點進來，返回回到原本那一頁', async ({ page }) => {
		const list = `${BASE}/${current}/men/matches/`;
		await page.goto(list);
		await openMoreSheet(page);
		await sheet(page).getByRole('link', { name: INSTALL_ROW }).click();
		await expect(page).toHaveURL(URL_PATH);
		await topBar(page).getByRole('link', { name: '返回' }).click();
		await expect(page).toHaveURL(list);
	});

	test('直接開網址，返回落在首頁', async ({ page }) => {
		await page.goto(URL_PATH);
		await topBar(page).getByRole('link', { name: '返回' }).click();
		await expect(page).toHaveURL(`${BASE}/${current}/men/`);
	});
});

test.describe('桌面', () => {
	test.use({ viewport: { width: 1440, height: 900 }, userAgent: UA.desktopChrome, isMobile: false, hasTouch: false });

	test('不支援推播時：h1 看得到；側欄有導覽、沒有目前頁、沒有組別與賽季切換；只有桌面那一句', async ({ page }) => {
		await withoutPush(page);
		await page.goto(URL_PATH);
		await expect(page.getByRole('heading', { level: 1, name: NAME })).toBeVisible();
		const side = page.getByRole('banner');
		const nav = side.getByRole('navigation', { name: '頁面' });
		await expect(nav.getByRole('link')).toHaveText(['首頁', '賽事', '戰績', '球隊', '球員', '排行榜', '歷史紀錄']);
		await expect(nav.getByRole('link', { name: '賽事' })).toHaveAttribute('href', `${BASE}/${current}/men/matches/`);
		await expect(side.locator('[aria-current]')).toHaveCount(0);
		await expect(side.getByRole('navigation', { name: '組別' })).toHaveCount(0);
		await expect(side.getByRole('navigation', { name: '賽季' })).toHaveCount(0);
		await expect(page.getByText('用手機瀏覽器開啟本站，就能把 VStat 企排加到主畫面。')).toBeVisible();
		await expect(steps(page)).toHaveCount(0);
		await expect(installButton(page)).toHaveCount(0);
	});

	test('支援推播時沒有桌面那一句', async ({ page }) => {
		await page.goto(URL_PATH);
		await expect(page.getByRole('switch')).toHaveCount(2);
		await expect(page.getByText('用手機瀏覽器開啟本站')).toHaveCount(0);
	});

	test('桌面 Chrome 自己的安裝事件不擋', async ({ page }) => {
		await page.goto(URL_PATH);
		expect(await fireInstallPrompt(page)).toBe(false);
		await expect(installButton(page)).toHaveCount(0);
	});
});

test.describe('iPhone Safari', () => {
	test.use({ userAgent: UA.iphoneSafari });

	test('三個步驟依序出現，沒有安裝鈕', async ({ page }) => {
		await page.goto(URL_PATH);
		await expect(steps(page)).toHaveText(['點分享鈕', '選「加入主畫面」', '點「加入」']);
		await expect(installButton(page)).toHaveCount(0);
	});
});

const inApp = [
	['iPhone 的 LINE', UA.iphoneLine, '請用 Safari 開啟'],
	['iPhone 的 Facebook', UA.iphoneFacebook, '請用 Safari 開啟'],
	['Android 的 LINE', UA.androidLine, '請用 Chrome 開啟'],
	['Android 的 Facebook', UA.androidFacebook, '請用 Chrome 開啟']
] as const;

for (const [name, userAgent, text] of inApp) {
	test.describe(name, () => {
		test.use({ userAgent });

		test(`寫「${text}」，沒有步驟與安裝鈕`, async ({ page }) => {
			await page.goto(URL_PATH);
			await expect(page.getByText(text)).toBeVisible();
			await expect(steps(page)).toHaveCount(0);
			await expect(installButton(page)).toHaveCount(0);
		});
	});
}

test.describe('Android Chrome', () => {
	test('事件來之前沒有安裝鈕；事件來了按鈕出現，連點兩下只叫 prompt 一次，結束後按鈕消失', async ({ page }) => {
		await page.goto(URL_PATH);
		await expect(page.getByText('請從 Chrome 的選單把本站加到主畫面')).toBeVisible();
		await expect(installButton(page)).toHaveCount(0);
		expect(await fireInstallPrompt(page)).toBe(true);
		await installButton(page).click();
		await installButton(page).click();
		expect(await promptCalls(page)).toBe(1);
		await expect(installButton(page)).toBeVisible();
		await finishChoice(page, 0);
		await expect(installButton(page)).toHaveCount(0);
	});

	test('取消後瀏覽器再送來的事件讓安裝鈕回來；安裝完成後按鈕收掉', async ({ page }) => {
		await page.goto(URL_PATH);
		await fireInstallPrompt(page);
		await installButton(page).click();
		await finishChoice(page, 0);
		await expect(installButton(page)).toHaveCount(0);
		await fireInstallPrompt(page);
		await expect(installButton(page)).toBeVisible();
		await page.evaluate(() => window.dispatchEvent(new Event('appinstalled')));
		await expect(installButton(page)).toHaveCount(0);
	});

	test('新的事件比上一個的結果早到：按鈕留著，再點用的是新的事件', async ({ page }) => {
		await page.goto(URL_PATH);
		await fireInstallPrompt(page);
		await installButton(page).click();
		await fireInstallPrompt(page);
		await finishChoice(page, 0);
		await installButton(page).click();
		expect(await promptCalls(page)).toBe(2);
	});

	test('不主動跳出：首頁收到事件，沒有安裝鈕，瀏覽器自己的橫條被擋下', async ({ page }) => {
		await page.goto(`${BASE}/${current}/men/`);
		expect(await fireInstallPrompt(page)).toBe(true);
		await expect(installButton(page)).toHaveCount(0);
	});
});

test.describe('獨立視窗', () => {
	test('不支援推播時顯示獨立視窗那一句，沒有步驟與安裝鈕', async ({ page }) => {
		await asStandalone(page);
		await withoutPush(page);
		await page.goto(URL_PATH);
		await expect(page.getByText('已經從主畫面開啟 VStat 企排，不用再加一次。')).toBeVisible();
		await expect(steps(page)).toHaveCount(0);
		await expect(installButton(page)).toHaveCount(0);
	});
});

test.describe('「更多」面板', () => {
	test('瀏覽器分頁內通知之後多一列「加到主畫面」', async ({ page }) => {
		await page.goto(`${BASE}/${current}/men/`);
		await openMoreSheet(page);
		await expect(sheet(page).getByRole('link')).toHaveText(['球員', '排行榜', '歷史紀錄', NAME, INSTALL_ROW]);
		await expect(sheet(page).getByRole('link', { name: INSTALL_ROW })).toHaveAttribute('href', URL_PATH);
	});

	test('獨立視窗內沒有「加到主畫面」那一列，通知還在', async ({ page }) => {
		await asStandalone(page);
		await page.goto(`${BASE}/${current}/men/`);
		await openMoreSheet(page);
		await expect(sheet(page).getByRole('link')).toHaveText(['球員', '排行榜', '歷史紀錄', NAME]);
	});

	test('在這一頁時，「通知」那一列是目前所在，「加到主畫面」那一列不是', async ({ page }) => {
		await page.goto(URL_PATH);
		await openMoreSheet(page);
		await expect(sheet(page).getByRole('link', { name: NAME })).toHaveAttribute('aria-current', 'page');
		await expect(sheet(page).getByRole('link', { name: INSTALL_ROW })).not.toHaveAttribute('aria-current');
	});
});
