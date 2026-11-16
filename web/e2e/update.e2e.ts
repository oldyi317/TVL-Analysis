import { readFileSync } from 'node:fs';
import { type Page } from '@playwright/test';
import { BASE, asStandalone, currentSeason, expect, loadAndWaitControlled, test as base } from './helpers.js';
import { startSite, type Site } from './site-server.js';

const { version } = JSON.parse(readFileSync('build/_app/version.json', 'utf8'));
const LABEL = '有新資料，點此更新';
const STANDINGS = `/${currentSeason}/men/standings/`;
// 新版要重新預存整批檔案，多個測試同時跑時比預設的 5 秒久
const SLOW = 20_000;
const standingsUrl = (site: Site) => `${site.origin}${BASE}${STANDINGS}`;

const test = base.extend<{ site: Site }>({
	site: async ({}, use) => {
		const site = await startSite(BASE, version);
		await use(site);
		await site.close();
	}
});

test.use({ serviceWorkers: 'allow' });

const updateBar = (page: Page) => page.getByRole('status').getByRole('button', { name: LABEL });
const mark = (page: Page) => page.evaluate(() => void ((window as unknown as { mark: number }).mark = 1));
const marked = (page: Page) => page.evaluate(() => (window as unknown as { mark?: number }).mark === 1);
const precaches = (page: Page) =>
	page.evaluate(async () => (await caches.keys()).filter((key) => key.startsWith('precache-')).sort());

// 頁面自己 location.reload() 時 helpers 的 hydration 等待不會介入：記號不見而且已 hydrate 就是重載完成；
// 重載當下 evaluate 會因文件被換掉而丟錯，算還沒完成
const reloadedByPage = (page: Page) => async () => {
	try {
		return !(await marked(page)) && (await page.locator('html[data-hydrated]').count()) === 1;
	} catch {
		return false;
	}
};

test('啟動時偵測到新版，抓完後出現更新條，頁面沒有被換掉、新版停在等待', async ({ page, site }) => {
	const url = standingsUrl(site);
	await loadAndWaitControlled(page, url);
	site.deploy();
	await page.reload();
	await mark(page);
	await expect(updateBar(page)).toBeVisible({ timeout: SLOW });
	expect(await marked(page)).toBe(true);
	expect(page.url()).toBe(url);
	expect(await page.evaluate(async () => (await navigator.serviceWorker.ready).waiting?.state)).toBe('installed');
	expect(await page.evaluate(() => navigator.serviceWorker.controller?.state)).toBe('activated');
	expect(await precaches(page)).toEqual([`precache-${version}`, `precache-${site.next}`]);
});

test('新版的檔案抓完才出現更新條，預存內容與舊版一致', async ({ page, site }) => {
	await loadAndWaitControlled(page, standingsUrl(site));
	const held = site.hold('/shell.html');
	site.deploy();
	await page.reload();
	await held.arrived;
	await expect
		.poll(() => page.evaluate(async () => (await navigator.serviceWorker.ready).installing !== null))
		.toBe(true);
	await expect(updateBar(page)).toHaveCount(0);
	held.release();
	await expect(updateBar(page)).toBeVisible({ timeout: SLOW });
	const urls = (key: string) =>
		page.evaluate(
			async (name) => (await (await caches.open(name)).keys()).map((r) => new URL(r.url).pathname).sort(),
			key
		);
	const fresh = await urls(`precache-${site.next}`);
	expect(fresh.length).toBeGreaterThan(1);
	expect(fresh).toEqual(await urls(`precache-${version}`));
});

async function showUpdateBar(page: Page, site: Site, path = STANDINGS) {
	const url = `${site.origin}${BASE}${path}`;
	await loadAndWaitControlled(page, url);
	site.deploy();
	await page.reload();
	await mark(page);
	await expect(updateBar(page)).toBeVisible({ timeout: SLOW });
	return url;
}

for (const how of ['鍵盤', '滑鼠'] as const) {
	test(`${how}按更新條：頁面重載、停在同一網址、只剩新版的預存`, async ({ page, site }) => {
		const url = await showUpdateBar(page, site);
		const bar = updateBar(page);
		if (how === '鍵盤') {
			await bar.focus();
			await page.keyboard.press('Enter');
		} else {
			await bar.click();
		}
		await expect.poll(reloadedByPage(page), { timeout: SLOW }).toBe(true);
		expect(page.url()).toBe(url);
		expect(await precaches(page)).toEqual([`precache-${site.next}`]);
		await expect(updateBar(page)).toHaveCount(0);
	});
}

test('不按更新條：重新整理仍是舊版，關閉重開才換新版', async ({ page, site, context }) => {
	const url = await showUpdateBar(page, site);
	await page.reload();
	await expect(updateBar(page)).toBeVisible({ timeout: SLOW });
	expect(await precaches(page)).toEqual([`precache-${version}`, `precache-${site.next}`]);
	await page.close();
	const reopened = await context.newPage();
	await reopened.goto(url);
	await expect.poll(() => precaches(reopened), { timeout: SLOW }).toEqual([`precache-${site.next}`]);
	await expect(updateBar(reopened)).toHaveCount(0);
});

// 新版一啟用就刪掉舊版的預存，還開著的舊頁面之後要載入的程式檔會拿不到，所以一起重載
test('兩個分頁：在一個分頁按更新條，另一個分頁也重載到新版、停在自己的網址', async ({ page, site, context }) => {
	await showUpdateBar(page, site);
	const other = await context.newPage();
	const otherUrl = `${site.origin}${BASE}/${currentSeason}/men/`;
	await other.goto(otherUrl);
	await mark(other);
	await expect(updateBar(other)).toBeVisible({ timeout: SLOW });
	await updateBar(page).click();
	await expect.poll(reloadedByPage(other), { timeout: SLOW }).toBe(true);
	expect(other.url()).toBe(otherUrl);
	await expect(updateBar(other)).toHaveCount(0);
});

// headless 不會真的變 hidden：覆寫 visibilityState 再送事件
const setVisibility = (page: Page, value: 'hidden' | 'visible') =>
	page.evaluate((state) => {
		Object.defineProperty(document, 'visibilityState', { value: state, configurable: true });
		document.dispatchEvent(new Event('visibilitychange'));
	}, value);

async function leaveAndReturn(page: Page, away: string) {
	await setVisibility(page, 'hidden');
	await page.clock.fastForward(away);
	await setVisibility(page, 'visible');
}

// 第一次進站伺服器會收到兩次腳本請求：註冊一次、啟動檢查一次。等兩次都到了再往下，之後多出來的才算得清楚是誰發的
const FIRST_VISIT_REQUESTS = 2;
const firstVisitSettled = (site: Site) =>
	expect.poll(() => site.scriptRequests.length).toBe(FIRST_VISIT_REQUESTS);

// 假時鐘要在頁面載入前裝好：載入後才裝，啟動時建立的計時器不歸它管
async function openStandalone(page: Page, site: Site) {
	await asStandalone(page);
	await page.clock.install();
	const url = standingsUrl(site);
	await loadAndWaitControlled(page, url);
	await mark(page);
	return url;
}

test('沒有新版：獨立視窗離開超過 30 分鐘回來只再檢查一次、不輪詢，什麼都不跳也不換頁', async ({ page, site }) => {
	await openStandalone(page, site);
	await firstVisitSettled(site);
	await page.clock.fastForward('03:00:00');
	await leaveAndReturn(page, '31:00');
	await expect.poll(() => site.scriptRequests.length).toBe(FIRST_VISIT_REQUESTS + 1);
	// 檢查是排隊依序做的：再排一次並等它做完，計時器若偷發過檢查，這時一定已經到伺服器
	await page.evaluate(async () => {
		await (await navigator.serviceWorker.ready).update();
	});
	expect(site.scriptRequests.length).toBe(FIRST_VISIT_REQUESTS + 2);
	await expect(updateBar(page)).toHaveCount(0);
	expect(await marked(page)).toBe(true);
});

test('獨立視窗離開超過 30 分鐘回來時沒有新版：之後才由別處發現的新版只出現更新條，不自動換', async ({ page, site }) => {
	await openStandalone(page, site);
	await firstVisitSettled(site);
	await leaveAndReturn(page, '31:00');
	await expect.poll(() => site.scriptRequests.length).toBe(FIRST_VISIT_REQUESTS + 1);
	site.deploy();
	// 不是這一頁回到前景發的檢查，例如同一個站的另一個分頁發的
	await page.evaluate(async () => {
		await (await navigator.serviceWorker.ready).update();
	});
	await expect(updateBar(page)).toBeVisible({ timeout: SLOW });
	expect(await marked(page)).toBe(true);
});

test('檢查新版繞過 HTTP 快取：伺服器回 max-age=600，重新整理後檢查請求仍到伺服器', async ({ page, site }) => {
	await loadAndWaitControlled(page, standingsUrl(site));
	expect(await page.evaluate(async () => (await navigator.serviceWorker.ready).updateViaCache)).toBe('none');
	await firstVisitSettled(site);
	await page.reload();
	// 不斷言剛好多一次：Chromium 在頁面載入完、網路靜下來後自己也會再檢查一次
	await expect.poll(() => site.scriptRequests.length).toBeGreaterThan(FIRST_VISIT_REQUESTS);
	expect(site.scriptRequests.at(-1)!['cache-control']).toBe('max-age=0');
});

test('獨立視窗離開超過 30 分鐘再回來：有新版就自動換，停在原本那一頁', async ({ page, site }) => {
	const url = await openStandalone(page, site);
	site.deploy();
	await leaveAndReturn(page, '31:00');
	await expect.poll(reloadedByPage(page), { timeout: SLOW }).toBe(true);
	expect(page.url()).toBe(url);
	expect(await precaches(page)).toEqual([`precache-${site.next}`]);
	await expect(updateBar(page)).toHaveCount(0);
});

test('獨立視窗離開未滿 30 分鐘：只出現更新條，不換頁', async ({ page, site }) => {
	await openStandalone(page, site);
	site.deploy();
	await leaveAndReturn(page, '29:00');
	await expect(updateBar(page)).toBeVisible({ timeout: SLOW });
	expect(await marked(page)).toBe(true);
});

test('瀏覽器分頁離開再久也只出現更新條，不自動換', async ({ page, site }) => {
	await page.clock.install();
	await loadAndWaitControlled(page, standingsUrl(site));
	await mark(page);
	site.deploy();
	await leaveAndReturn(page, '05:00:00');
	await expect(updateBar(page)).toBeVisible({ timeout: SLOW });
	expect(await marked(page)).toBe(true);
	expect(await precaches(page)).toEqual([`precache-${version}`, `precache-${site.next}`]);
});

test('獨立視窗更新條已出現後又離開超過 30 分鐘：回來時自動換', async ({ page, site }) => {
	const url = await openStandalone(page, site);
	site.deploy();
	await leaveAndReturn(page, '29:00');
	await expect(updateBar(page)).toBeVisible({ timeout: SLOW });
	await leaveAndReturn(page, '31:00');
	await expect.poll(reloadedByPage(page), { timeout: SLOW }).toBe(true);
	expect(page.url()).toBe(url);
	expect(await precaches(page)).toEqual([`precache-${site.next}`]);
});

const box = async (locator: ReturnType<Page['locator']>) => (await locator.boundingBox())!;

test('手機寬度：更新條不被分頁列與安全區蓋住，按鈕夠大、點得到', async ({ page, site }) => {
	await showUpdateBar(page, site);
	const button = updateBar(page);
	const tabs = page.getByRole('navigation', { name: '主要分頁' });
	const viewport = page.viewportSize()!;
	const check = async (side: number) => {
		const b = await box(button);
		const t = await box(tabs);
		expect(b.y + b.height).toBeLessThanOrEqual(t.y);
		expect(b.height).toBeGreaterThanOrEqual(44);
		expect(b.x).toBeGreaterThanOrEqual(side);
		expect(b.x + b.width).toBeLessThanOrEqual(viewport.width - side);
		const hit = await button.evaluate((el) => {
			const r = el.getBoundingClientRect();
			return document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2) === el;
		});
		expect(hit).toBe(true);
		return t.y;
	};
	const plainTabTop = await check(0);
	await page.addStyleTag({
		content: ':root { --safe-bottom: 34px; --safe-left: 20px; --safe-right: 20px; }'
	});
	await check(20);
	const safeTabTop = (await box(tabs)).y;
	expect(safeTabTop).toBeLessThan(plainTabTop);
});

test.describe('桌面寬度', () => {
	test.use({ viewport: { width: 1280, height: 800 } });

	test('更新條在視窗內、不與側欄重疊', async ({ page, site }) => {
		await showUpdateBar(page, site);
		const b = await box(updateBar(page));
		const side = await box(page.getByRole('banner'));
		expect(b.x).toBeGreaterThanOrEqual(side.x + side.width);
		expect(b.x + b.width).toBeLessThanOrEqual(1280);
		expect(b.y).toBeGreaterThanOrEqual(0);
		expect(b.y + b.height).toBeLessThanOrEqual(800);
	});
});
