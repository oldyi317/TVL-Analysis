// 用 Playwright 把 web/static/icons/*.svg 轉成 PNG（build 時不重產）：node prototypes/offline/make-icons.mjs
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const { chromium } = createRequire(root + 'web/package.json')('playwright');
const dir = root + 'web/static/icons/';
const jobs = [
	['icon.svg', 192, 'icon-192.png'],
	['icon.svg', 512, 'icon-512.png'],
	['icon-maskable.svg', 512, 'icon-512-maskable.png'],
	['icon.svg', 180, 'icon-180.png']
];
const browser = await chromium.launch();
for (const [src, size, out] of jobs) {
	const page = await browser.newPage({ viewport: { width: size, height: size } });
	const svg = readFileSync(dir + src, 'utf8');
	await page.setContent(`<body style="margin:0">${svg.replace('<svg ', `<svg width="${size}" height="${size}" style="display:block" `)}</body>`);
	writeFileSync(dir + out, await page.screenshot({ type: 'png', omitBackground: false }));
	await page.close();
	console.log('wrote', out);
}
await browser.close();
