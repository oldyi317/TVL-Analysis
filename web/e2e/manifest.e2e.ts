import { type APIRequestContext } from '@playwright/test';
import { BASE, currentSeason, expect, test } from './helpers.js';

type Manifest = {
	name: string;
	short_name: string;
	display: string;
	start_url: string;
	scope: string;
	id: string;
	background_color: string;
	theme_color: string;
	icons: { src: string; sizes: string; type: string; purpose?: string }[];
};

const getManifest = async (request: APIRequestContext) => {
	const res = await request.get(`${BASE}/manifest.webmanifest`);
	expect(res.status()).toBe(200);
	expect(res.headers()['content-type']).toContain('application/manifest+json');
	return (await res.json()) as Manifest;
};

test('manifest 內容', async ({ request }) => {
	const manifest = await getManifest(request);
	expect(manifest.name).toBe('VStat 企排');
	expect(manifest.short_name).toBe('VStat 企排');
	expect(manifest.display).toBe('standalone');
	expect(manifest.start_url).toBe(`${BASE}/`);
	expect(manifest.scope).toBe(`${BASE}/`);
	expect(manifest.id).toBe(`${BASE}/`);
	expect(manifest.background_color).toBe('#f3f4f9');
	expect(manifest.theme_color).toBe('#f3f4f9');
	expect(manifest.icons).toEqual([
		{ src: `${BASE}/icons/icon-192.png`, sizes: '192x192', type: 'image/png' },
		{ src: `${BASE}/icons/icon-512.png`, sizes: '512x512', type: 'image/png' },
		{ src: `${BASE}/icons/icon-maskable-512.png`, sizes: '512x512', type: 'image/png', purpose: 'maskable' }
	]);
	for (const key of ['orientation', 'screenshots', 'shortcuts']) {
		expect(manifest).not.toHaveProperty(key);
	}
});

test('圖示檔存在、尺寸與宣告相符，apple-touch-icon 不透明', async ({ page }) => {
	const manifest = await getManifest(page.request);
	const icons = [
		...manifest.icons.map((icon) => ({ src: icon.src, size: Number(icon.sizes.split('x')[0]) })),
		{ src: `${BASE}/icons/apple-touch-icon.png`, size: 180 }
	];
	await page.goto(`${BASE}/`);
	for (const { src, size } of icons) {
		const res = await page.request.get(src);
		expect(res.status(), src).toBe(200);
		expect(res.headers()['content-type'], src).toContain('image/png');
		const measured = await page.evaluate(async (url) => {
			const bitmap = await createImageBitmap(await (await fetch(url)).blob());
			const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
			const ctx = canvas.getContext('2d')!;
			ctx.drawImage(bitmap, 0, 0);
			const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
			let minAlpha = 255;
			for (let i = 3; i < data.length; i += 4) minAlpha = Math.min(minAlpha, data[i]);
			return { width: bitmap.width, height: bitmap.height, minAlpha };
		}, src);
		expect(measured.width, src).toBe(size);
		expect(measured.height, src).toBe(size);
		if (src.endsWith('apple-touch-icon.png')) expect(measured.minAlpha).toBe(255);
	}
});

for (const [name, path] of [
	['賽季組別頁', `${BASE}/${currentSeason}/men/`],
	['歷史紀錄頁', `${BASE}/records/men/`]
]) {
	test(`${name}的 head 有 manifest 與 apple-touch-icon 連結`, async ({ page }) => {
		await page.goto(path);
		// 水合後 asset() 在瀏覽器端會變成帶來源的絕對網址，所以比路徑
		const pathOf = async (rel: string) =>
			new URL((await page.locator(`link[rel="${rel}"]`).getAttribute('href'))!, page.url()).pathname;
		expect(await pathOf('manifest')).toBe(`${BASE}/manifest.webmanifest`);
		expect(await pathOf('apple-touch-icon')).toBe(`${BASE}/icons/apple-touch-icon.png`);
	});
}

test('從 manifest 的 start_url 開啟會到上次看的組別', async ({ page }) => {
	const manifest = await getManifest(page.request);
	await page.goto(`${BASE}/${currentSeason}/women/`);
	await expect.poll(() => page.evaluate(() => localStorage.getItem('tvl.group'))).toBe('women');
	await page.goto(manifest.start_url);
	await expect(page).toHaveURL(`${BASE}/${currentSeason}/women/`);
});
