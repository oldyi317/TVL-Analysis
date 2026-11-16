import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import Page from './+page.svelte';

describe('/notifications/ 頁（prerender 時不知道瀏覽器種類）', () => {
	const context = new Map([['__request__', { page: { url: new URL('/notifications/', 'http://localhost') } }]]);
	const { head, body } = render(Page, { context });

	it('title 是「加到主畫面｜VStat 企排」', () => {
		expect(head).toContain('<title>加到主畫面｜VStat 企排</title>');
	});

	it('只有一個 h1「加到主畫面」', () => {
		expect([...body.matchAll(/<h1[^>]*>([^<]*)<\/h1>/g)].map((m) => m[1])).toEqual(['加到主畫面']);
	});

	it.each(['請用', '安裝 VStat 企排', '點分享鈕', '已經從主畫面開啟', '用手機瀏覽器'])('不輸出引導字句「%s」', (text) => {
		expect(body).not.toContain(text);
	});
});
