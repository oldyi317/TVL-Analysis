import type { ChartSpec, Line } from './spec.js';

const HTML_ESCAPES: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

function escapeHtml(text: string): string {
	return text.replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);
}

function formatLines(lines: Line[]): string {
	return lines.map((line) => typeof line === 'string'
		? escapeHtml(line)
		: '<b>' + escapeHtml(line.bold) + '</b>'
	).join('<br/>');
}

export function tooltipOption<D>(tip: ChartSpec<D>['tooltip']) {
	if (tip.trigger === 'axis') {
		return {
			confine: true,
			trigger: tip.trigger,
			...(tip.axisPointer && { axisPointer: tip.axisPointer }),
			formatter: (params: { dataIndex: number }[]) => formatLines(tip.lines(params[0].dataIndex))
		};
	}
	return {
		confine: true,
		trigger: tip.trigger,
		formatter: ({ name, value, data }: { name: string; value: unknown; data: D }) =>
			formatLines(tip.lines({ name, value, data }))
	};
}
