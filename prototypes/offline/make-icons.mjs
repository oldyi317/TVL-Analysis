// 用 Playwright 把 prototypes/offline/icon-source.png（使用者生成的 1254×1254 原稿）縮成四張 PNG
// 放到 web/static/icons/（build 時不重產）：node prototypes/offline/make-icons.mjs
// 原稿不放 web/static：放進去會被 service worker 預存，首次下載多 1.2 MB。
// 原稿的主體約佔畫面中央 57%，已在 maskable 的安全區（中央 80%）內，所以 maskable 與一般版同圖。
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const { chromium } = createRequire(root + 'web/package.json')('playwright');
const src = 'data:image/png;base64,' + readFileSync(root + 'prototypes/offline/icon-source.png').toString('base64');
const dir = root + 'web/static/icons/';
const jobs = [
	[192, 'vstat-192.png'],
	[512, 'vstat-512.png'],
	[512, 'vstat-512-maskable.png'],
	[180, 'vstat-180.png']
];
const browser = await chromium.launch();
for (const [size, out] of jobs) {
	const page = await browser.newPage({ viewport: { width: size, height: size } });
	await page.setContent(`<body style="margin:0"><img src="${src}" width="${size}" height="${size}" style="display:block"></body>`);
	await page.locator('img').evaluate((img) => img.decode());
	writeFileSync(dir + out, await page.screenshot({ type: 'png', omitBackground: false }));
	await page.close();
	console.log('wrote', out);
}
await browser.close();
