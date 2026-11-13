import { createRawSnippet } from 'svelte';
import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import Layout from '../+layout.svelte';
import Page from './+page.svelte';

describe('404 頁', () => {
	const context = new Map([
		['__request__', { page: { url: new URL('/404', 'http://localhost') } }]
	]);
	const { head, body } = render(Page, { context });

	it('顯示「找不到這頁」並有回首頁連結', () => {
		expect(body).toContain('找不到這頁');
		expect(body).toMatch(/<a[^>]*href="\/"[^>]*>回首頁<\/a>/);
	});

	it('title 維持「…｜VStat 企排」格式', () => {
		expect(head).toContain('<title>找不到這頁｜VStat 企排</title>');
	});
});

describe('找不到的賽季網址：導覽連結留在本站', () => {
	const children = createRawSnippet(() => ({ render: () => '<p>找不到這頁</p>' }));

	function hrefsAt(path: string, routeId: string, params: Record<string, string>) {
		const url = new URL(path, 'https://vstat.tw');
		const context = new Map([['__request__', { page: { url, params, route: { id: routeId }, data: {} } }]]);
		const { body } = render(Layout, { props: { children } as never, context });
		const back = body.match(/<a[^>]*href="([^"]*)"[^>]*aria-label="返回"/)?.[1];
		return { url, back, hrefs: [...body.matchAll(/href="([^"]*)"/g)].map((m) => m[1]) };
	}

	function expectOnSite(url: URL, hrefs: string[]) {
		expect(hrefs.some((h) => new URL(h, url).host.includes('evil.example'))).toBe(false);
		for (const href of hrefs.filter((h) => h.startsWith('/'))) {
			expect(href).not.toMatch(/^\/[\\/]/);
			expect(new URL(href, url).origin).toBe(url.origin);
		}
	}

	it('賽季片段帶反斜線時，站內連結都留在本站，賽季片段已編碼', () => {
		const { url, hrefs } = hrefsAt('/%5Cevil.example/men/', '/[season]/[group]', {
			season: '\\evil.example',
			group: 'men'
		});
		expect(hrefs).toContain('/%5Cevil.example/men/standings/');
		expectOnSite(url, hrefs);
	});

	it('球隊詳情網址帶反斜線賽季時，返回連結也留在本站', () => {
		const { url, back, hrefs } = hrefsAt('/%5Cevil.example/men/teams/1/', '/[season]/[group]/teams/[team]', {
			season: '\\evil.example',
			group: 'men',
			team: '1'
		});
		expect(back).toBe('/%5Cevil.example/men/teams/');
		expectOnSite(url, hrefs);
	});

	it('不存在但無害的賽季片段，導覽、組別與賽季切換照舊指向原片段', () => {
		const { hrefs } = hrefsAt('/cup99/men/', '/[season]/[group]', { season: 'cup99', group: 'men' });
		for (const section of ['', 'matches/', 'standings/', 'teams/', 'players/', 'leaders/']) {
			expect(hrefs).toContain(`/cup99/men/${section}`);
		}
		expect(hrefs).toContain('/cup99/women/');
		expect(hrefs).toContain('/2025-26/men/');
	});

	it('不存在但無害的賽季片段，詳情頁返回連結照舊指向原片段', () => {
		const { back } = hrefsAt('/cup99/men/teams/1/', '/[season]/[group]/teams/[team]', {
			season: 'cup99',
			group: 'men',
			team: '1'
		});
		expect(back).toBe('/cup99/men/teams/');
	});

	it('實際賽季的導覽連結不變', () => {
		const { hrefs } = hrefsAt('/2025-26/women/', '/[season]/[group]', { season: '2025-26', group: 'women' });
		expect(hrefs).toContain('/2025-26/women/');
		expect(hrefs).toContain('/2025-26/women/standings/');
		expect(hrefs).toContain('/records/women/');
	});
});
