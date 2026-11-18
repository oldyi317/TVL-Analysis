import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import Page from './+page.svelte';

describe('/notifications/ 頁（prerender 時不知道瀏覽器種類與能力）', () => {
	const context = new Map([['__request__', { page: { url: new URL('/notifications/', 'http://localhost') } }]]);
	const { head, body } = render(Page, { context });

	it('title 是「通知｜VStat 企排」', () => {
		expect(head).toContain('<title>通知｜VStat 企排</title>');
	});

	it('只有一個 h1「通知」', () => {
		expect([...body.matchAll(/<h1[^>]*>([^<]*)<\/h1>/g)].map((m) => m[1])).toEqual(['通知']);
	});

	it.each([
		'請用',
		'安裝 VStat 企排',
		'點分享鈕',
		'已經從主畫面開啟',
		'用手機瀏覽器',
		'加到主畫面後才能開啟通知',
		'role="switch"',
		'開啟後，本站會存下'
	])('不輸出引導或開關字句「%s」', (text) => {
		expect(body).not.toContain(text);
	});
});
