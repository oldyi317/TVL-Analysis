// 只給圖表層（spec.ts）內部用；各圖交代圖例項目給 chartOption，不直接呼叫這裡。
// 圖例項目的數量與長度不固定（例如隊名）時，圖例在各圖寬下折成幾行不固定，grid.bottom 不能寫死。
// 這裡照 ECharts 圖例的折行規則，依項目名稱算出各圖寬下的行數與要留的高度。
// 尺寸取自 ECharts 圖例預設值；字寬一律以全形字計，算出的行數只會多、不會少。
import { LEGEND_FONT_SIZE } from './theme.js';

const ICON = 26;
const ITEM_GAP = 8;
const ITEM_HEIGHT = 14;
const PADDING = 5;

export type LegendTier = { minWidth: number; height: number };

function rowCount(widths: readonly number[], max: number): number {
	let rows = 0;
	let used = 0;
	for (const width of widths) {
		const next = used + ITEM_GAP + width;
		if (rows > 0 && next <= max) {
			used = next;
			continue;
		}
		rows++;
		used = width;
	}
	return rows;
}

// 行數只在「連續幾項排成一行剛好放得下」的寬度改變，由窄到寬列出這些寬度
function runWidths(widths: readonly number[]): number[] {
	const runs = new Set<number>();
	widths.forEach((_, from) => {
		let run = -ITEM_GAP;
		for (const width of widths.slice(from)) {
			run += ITEM_GAP + width;
			runs.add(run);
		}
	});
	return [...runs].sort((x, y) => x - y);
}

// 由窄到寬列出圖例行數改變的容器寬度與該寬度起的圖例高度；第一筆從 0 起，是每項各佔一行
export function legendTiers(names: readonly string[]): LegendTier[] {
	const widths = names.map((name) => [...name].length * LEGEND_FONT_SIZE + ICON);
	const height = (rows: number) => rows * ITEM_HEIGHT + (rows - 1) * ITEM_GAP + PADDING * 2;
	const tiers = [{ minWidth: 0, height: height(names.length) }];
	let fewest = names.length;
	for (const run of runWidths(widths)) {
		const rows = rowCount(widths, run);
		if (rows >= fewest) continue;
		fewest = rows;
		tiers.push({ minWidth: run + PADDING * 2, height: height(rows) });
	}
	return tiers;
}
