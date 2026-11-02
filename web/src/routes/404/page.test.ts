import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import Page from './+page.svelte';

describe('404 頁', () => {
	const context = new Map([
		['__request__', { page: { url: new URL('/TVL-Analysis/404', 'http://localhost') } }]
	]);
	const { head, body } = render(Page, { context });

	it('顯示「找不到這頁」並有回首頁連結', () => {
		expect(body).toContain('找不到這頁');
		expect(body).toMatch(/<a[^>]*href="\/TVL-Analysis\/"[^>]*>回首頁<\/a>/);
	});

	it('title 維持「…｜TVL 數據」格式', () => {
		expect(head).toContain('<title>找不到這頁｜TVL 數據</title>');
	});
});
