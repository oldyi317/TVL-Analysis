// 圖例項目的數量與長度不固定（例如隊名）時，圖例在各圖寬下折成幾行不固定，grid.bottom 不能寫死。
// 這裡照 ECharts 圖例的折行規則，依項目名稱算出各圖寬下的行數，回傳替圖例留高度的 media 設定。
// 尺寸取自 ECharts 圖例預設值；字寬一律以全形字計，算出的行數只會多、不會少。
import { LEGEND_FONT_SIZE } from './theme.js';

const ICON = 26;
const ITEM_GAP = 8;
const ITEM_HEIGHT = 14;
const PADDING = 5;

export type LegendMedia = { query?: { minWidth: number }; option: { grid: { bottom: number } } };

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

// gap：圖例上緣到繪圖區（含刻度標籤）之間要留的高度，例如放橫軸軸名
export function bottomLegendMedia(names: readonly string[], gap: number): LegendMedia[] {
	const widths = names.map((name) => [...name].length * LEGEND_FONT_SIZE + ICON);
	const option = (rows: number) => ({
		grid: { bottom: rows * ITEM_HEIGHT + (rows - 1) * ITEM_GAP + PADDING * 2 + gap }
	});
	const tiers: LegendMedia[] = [];
	let fewest = names.length;
	for (const run of runWidths(widths)) {
		const rows = rowCount(widths, run);
		if (rows >= fewest) continue;
		fewest = rows;
		tiers.push({ query: { minWidth: run + PADDING * 2 }, option: option(rows) });
	}
	// ECharts 依序套用所有符合的 query，最寬的排最後；都不符合才用沒有 query 的那一筆
	return [...tiers, { option: option(names.length) }];
}
