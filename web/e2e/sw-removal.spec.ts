import { readFileSync } from 'node:fs';
import { type Page } from '@playwright/test';
import { BASE, currentSeason, expect, loadAndWaitControlled, setVisibility, test as base } from './helpers.js';
import { startSite, type Site } from './site-server.js';

const { version } = JSON.parse(readFileSync('build/_app/version.json', 'utf8'));
const REMOVED = 'build-remove';
const SLOW = 20_000;
const PAGE = `${BASE}/${currentSeason}/men/standings/`;

const test = base.extend<{ site: Site }>({
	site: async ({}, use) => {
		const site = await startSite(BASE, version);
		await use(site);
		await site.close();
	}
});

test.use({ serviceWorkers: 'allow' });

const cacheNames = (page: Page) => page.evaluate(() => caches.keys());
const registered = (page: Page) =>
	page.evaluate(async () => (await navigator.serviceWorker.getRegistration()) !== undefined);

// 頁面正在重新載入時 evaluate 會丟錯：輪詢時當作還沒好（null 不等於任何期望值）
const orNull = async <T>(read: () => Promise<T>) => {
	try {
		return await read();
	} catch {
		return null;
	}
};

async function enter(page: Page, site: Site) {
	await loadAndWaitControlled(page, `${site.origin}${PAGE}`);
	expect((await cacheNames(page)).length).toBeGreaterThan(0);
	expect(await registered(page)).toBe(true);
}

async function expectRemoved(page: Page) {
	await expect.poll(() => orNull(() => cacheNames(page)), { timeout: SLOW }).toEqual([]);
	await expect.poll(() => orNull(() => registered(page)), { timeout: SLOW }).toBe(false);
}

test('部署自我移除版後再開站：cache 清空、註銷、頁面不再註冊，斷網就載入失敗', async ({ page, site, context }) => {
	await enter(page, site);
	site.deploy(REMOVED);
	await page.reload();
	await expectRemoved(page);
	await page.reload();
	// 頁面若還會註冊，是在啟動時發出的：等網路靜下來再看
	await page.waitForLoadState('networkidle');
	expect(await page.evaluate(() => navigator.serviceWorker.controller)).toBeNull();
	expect(await registered(page)).toBe(false);
	expect(await cacheNames(page)).toEqual([]);
	await context.setOffline(true);
	await expect(page.reload()).rejects.toThrow();
});

test('部署時還開著的舊分頁：回到前景後自己重新載入，之後 cache 清空、沒有註冊', async ({ page, site }) => {
	await enter(page, site);
	site.deploy(REMOVED);
	const reloaded = page.waitForEvent('load', { timeout: SLOW });
	await setVisibility(page, 'hidden');
	await setVisibility(page, 'visible');
	await reloaded;
	await expectRemoved(page);
});

// 部署後 10 分鐘內瀏覽器的 HTTP 快取還可能拿出舊頁面，舊頁面會照正式版的寫法再註冊一次
test('已經清乾淨的裝置又被舊頁面註冊：裝好就再註銷，不留 cache', async ({ page, site }) => {
	site.deploy(REMOVED);
	await page.goto(`${site.origin}${PAGE}`);
	await page.waitForLoadState('networkidle');
	expect(await registered(page)).toBe(false);
	// 預先放一個 cache：它被清掉就表示自我移除版真的裝起來跑過
	await page.evaluate(async (scope) => {
		await caches.open('leftover');
		await navigator.serviceWorker.register(`${scope}service-worker.js`, {
			scope,
			type: 'module',
			updateViaCache: 'none'
		});
	}, `${BASE}/`);
	await expectRemoved(page);
	expect(await page.evaluate(() => navigator.serviceWorker.controller)).toBeNull();
});
