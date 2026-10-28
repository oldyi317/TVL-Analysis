import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import siteJson from '#lib/data/site.json';
import { formatTaipeiMinute } from '#lib/datetime.js';
import SiteFooter from './SiteFooter.svelte';

describe('SiteFooter', () => {
	const html = render(SiteFooter).body;

	it('顯示資料更新時間（台北時間到分）', () => {
		expect(html).toContain('資料更新時間');
		expect(html).toContain(formatTaipeiMinute(siteJson.generated_at));
	});

	it('有 Issues 與官網外部連結，開新分頁並帶 noopener', () => {
		expect(html).toMatch(
			/<a[^>]*href="https:\/\/github\.com\/oldyi317\/TVL-Analysis\/issues"[^>]*>回報錯誤或許願<\/a>/
		);
		expect(html).toMatch(/<a[^>]*href="https:\/\/tvl\.ctvba\.org\.tw[^"]*"[^>]*>TVL 官網<\/a>/);
		const external = html.match(/<a [^>]*href="https?:[^>]*>/g) ?? [];
		expect(external.length).toBeGreaterThanOrEqual(3);
		for (const tag of external) {
			expect(tag).toContain('target="_blank"');
			expect(tag).toMatch(/rel="[^"]*noopener[^"]*"/);
		}
	});

	it('既有兩段文案一字不改', () => {
		expect(html).toContain(
			'資料來源：TVL 企業排球聯賽官網與其技術統計系統。本站為球迷自製的非官方網站，數據僅供參考，與聯賽及各球隊無關。'
		);
		expect(html).toContain('隊徽與球隊名稱之商標屬各球隊與聯賽所有。程式碼以 MIT 授權公開於');
	});
});
