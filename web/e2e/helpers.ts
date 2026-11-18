import { readFileSync } from 'node:fs';
import { test as base, expect, type Page } from '@playwright/test';
import { KINDS, pickSeason, type Kind } from './pick-season.js';

export { expect };

// 所有 e2e 都從這裡匯入 test：載入新文件的 goto／reload 等到根 layout 留下 hydration 記號才返回，
// 進頁後的第一個互動就不會早於 hydration
export const test = base.extend({
	page: async ({ page }, use) => {
		const waitHydrated =
			<A extends unknown[], R>(navigate: (...args: A) => Promise<R>) =>
			async (...args: A) => {
				const response = await navigate(...args);
				await page.locator('html[data-hydrated]').waitFor({ state: 'attached' });
				return response;
			};
		page.goto = waitHydrated(page.goto.bind(page));
		page.reload = waitHydrated(page.reload.bind(page));
		await use(page);
	}
});

export const BASE = '';
export const OLD_CUP = 20;
export const OLD_SEASON = '2024-25';
export const readJson = (name: string) =>
	JSON.parse(readFileSync(`src/lib/data/${name}.json`, 'utf8'));
export const seasonsData = readJson('seasons');
export const currentSeason: string = seasonsData.seasons.find(
	(s: { cup_id: number }) => s.cup_id === seasonsData.current_cup_id
).name;
export const currentMatches = (group: string): { date: string }[] =>
	readJson('matches').matches.find(
		(g: { cup_id: number; group: string }) => g.cup_id === seasonsData.current_cup_id && g.group === group
	)?.matches ?? [];

// 讀匯出 JSON 的薄包裝：給 spec 取「最近一個有該資料的賽季」
export const findSeason = (kind: Kind, group = 'men') => {
	const { file, key } = KINDS[kind];
	return pickSeason(kind, group, {
		rows: readJson(file)[key ?? file],
		seasons: seasonsData.seasons,
		currentCupId: seasonsData.current_cup_id
	});
};

// 假裝是以獨立視窗開啟：只改 display-mode: standalone 的查詢，其他照原本；要在載入頁面之前呼叫
export async function asStandalone(page: Page) {
	await page.addInitScript(() => {
		const original = window.matchMedia.bind(window);
		window.matchMedia = (query) => original(query.includes('display-mode: standalone') ? 'all' : query);
	});
}

// 線上載入一頁，等 service worker 預存完、啟用並接手
export async function loadAndWaitControlled(page: Page, url: string) {
	await page.goto(url);
	await page.waitForFunction(() => navigator.serviceWorker.controller !== null);
}

// headless 不會真的變 hidden：覆寫 visibilityState 再送事件
export const setVisibility = (page: Page, value: 'hidden' | 'visible') =>
	page.evaluate((state) => {
		Object.defineProperty(document, 'visibilityState', { value: state, configurable: true });
		document.dispatchEvent(new Event('visibilitychange'));
	}, value);

export async function openMoreSheet(page: Page) {
	await page.getByRole('navigation', { name: '主要分頁' }).getByRole('button', { name: '更多' }).click();
	await expect(page.getByRole('dialog', { name: '更多' })).toBeVisible();
}

// 手機頂部列；banner 在桌面是側欄，所以組別連結的字尾「組」可有可無（手機「女子」、桌面「女子組」）
export const topBar = (page: Page) => page.getByRole('banner');
export const groupLink = (page: Page, label: '男子' | '女子') =>
	topBar(page).getByRole('navigation', { name: '組別' }).getByRole('link', { name: new RegExp(`^${label}組?$`) });
export const seasonButton = (page: Page) => topBar(page).getByRole('button', { name: /切換賽季/ });
export const seasonSheet = (page: Page) => page.getByRole('dialog', { name: '賽季' });

export async function openSeasonSheet(page: Page) {
	await seasonButton(page).click();
	await expect(seasonSheet(page)).toBeVisible();
}

// 在頂部列點組別
export async function switchGroup(page: Page, label: '男子' | '女子', url: string) {
	await groupLink(page, label).click();
	await expect(page).toHaveURL(url);
}

// 開賽季面板選賽季；選完面板會關閉
export async function switchSeason(page: Page, season: string, url: string) {
	await openSeasonSheet(page);
	await seasonSheet(page).getByRole('link', { name: season, exact: true }).click();
	await expect(page).toHaveURL(url);
}
