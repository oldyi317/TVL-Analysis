import { describe, expect, it } from 'vitest';
import { bottomLegendMedia } from './legend.js';

// 寬度以 14px 的圖例字計：每項 = 字數 × 14 + 26，項距 8，圖例左右各留 5；一行高 24，每多一行加 22
describe('bottomLegendMedia', () => {
	const men = ['屏東台電', '雲林美津濃', '臺北國北獅', '桃園臺產', '獅子王'];
	const women = ['臺北鯨華', '高雄台電', '新北中纖', '義力營造'];
	const bottomAt = (names: string[], width: number) =>
		bottomLegendMedia(names, 0)
			.filter((m) => m.query && m.query.minWidth <= width)
			.at(-1)!.option.grid.bottom;

	it('五隊：圖例隨圖寬折成 1 到 5 行，底部留圖例高度加上指定的間距', () => {
		expect(bottomLegendMedia(men, 40)).toEqual([
			{ query: { minWidth: 168 }, option: { grid: { bottom: 130 } } },
			{ query: { minWidth: 196 }, option: { grid: { bottom: 108 } } },
			{ query: { minWidth: 272 }, option: { grid: { bottom: 86 } } },
			{ query: { minWidth: 466 }, option: { grid: { bottom: 64 } } },
			{ option: { grid: { bottom: 152 } } }
		]);
	});

	it('320px 手機的圖寬 254：五隊留三行、四隊留兩行；375px 手機的圖寬 309：都留兩行', () => {
		expect([bottomAt(men, 254), bottomAt(women, 254)]).toEqual([68, 46]);
		expect([bottomAt(men, 309), bottomAt(women, 309)]).toEqual([46, 46]);
	});

	it('英數字也以一個全形字的寬度計，寧可多留不少留', () => {
		expect(bottomLegendMedia(['臺北Conti', '連莊'], 0)).toEqual([
			{ query: { minWidth: 196 }, option: { grid: { bottom: 24 } } },
			{ option: { grid: { bottom: 46 } } }
		]);
	});

	it('只有一項時不分寬度都留一行', () => {
		expect(bottomLegendMedia(['屏東台電'], 8)).toEqual([{ option: { grid: { bottom: 32 } } }]);
	});
});
