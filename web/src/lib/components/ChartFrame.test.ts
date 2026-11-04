import { createRawSnippet } from 'svelte';
import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import ChartFrame from './ChartFrame.svelte';

const table = createRawSnippet(() => ({ render: () => '<table aria-label="表"></table>' }));
const frame = (props: { option: object | null; conclusion: string | null; games?: number | { name: string; games: number }[] }) =>
	render(ChartFrame, {
		props: { title: '球隊指標對照組內各隊', games: props.games ?? 3, emptyTitle: '本季尚無單場數據', table, ...props }
	}).body;

describe('ChartFrame', () => {
	it('有一句話結論時顯示結論，無障礙文字也包含結論', () => {
		const body = frame({ option: {}, conclusion: '接發球到位率 第 1 / 6' });
		expect(body).toContain('<p class="conclusion');
		expect(body).toContain('aria-label="球隊指標對照組內各隊：接發球到位率 第 1 / 6"');
		expect(body).toContain('已賽 3 場');
	});

	it('沒有結論時無障礙文字改用圖名加已賽場數', () => {
		const body = frame({ option: {}, conclusion: null, games: 2 });
		expect(body).not.toContain('class="conclusion');
		expect(body).toContain('aria-label="球隊指標對照組內各隊，已賽 2 場"');
	});

	it('以表格檢視按鈕預設收合，有 aria-expanded 與 aria-controls', () => {
		const body = frame({ option: {}, conclusion: null });
		expect(body).toMatch(/<button[^>]*aria-expanded="false"[^>]*aria-controls="[^"]+-table"[^>]*>以表格檢視/);
	});

	it('沒有數據時顯示空狀態，不畫圖也不給表格切換', () => {
		const body = frame({ option: null, conclusion: null });
		expect(body).toContain('本季尚無單場數據');
		expect(body).not.toContain('role="img"');
		expect(body).not.toContain('以表格檢視');
		expect(body).not.toContain('已賽');
	});

	it('兩隊對照時分列兩隊已賽場數，場數相同時合併', () => {
		const two = (a: number, b: number) =>
			frame({ option: {}, conclusion: null, games: [{ name: '台電', games: a }, { name: '鳳凰', games: b }] });
		expect(two(3, 2)).toContain('台電 已賽 3 場・鳳凰 已賽 2 場');
		expect(two(3, 2)).toContain('aria-label="球隊指標對照組內各隊，台電 已賽 3 場・鳳凰 已賽 2 場"');
		expect(two(3, 3)).toContain('>已賽 3 場<');
	});
});
