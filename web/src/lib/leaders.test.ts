import { describe, expect, it } from 'vitest';
import type { VolumeRate } from '#lib/data/types.js';
import { seasons, seasonTeam, teams } from '#lib/site.js';
import {
	formatThroughDate,
	toVolumeRateView,
	visibleRows,
	VOLUME_RATE_TECHNIQUES,
	volumeRateNote,
	volumeRateOption
} from './leaders.js';

const rows = (ranks: number[]) => ranks.map((rank, i) => ({ rank, player_id: i + 1 }));

describe('排行榜前 10 名截斷', () => {
	it('超過 10 人只留前 10 名', () => {
		const shown = visibleRows(rows([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
		expect(shown.map((r) => r.rank)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
	});

	it('第 10 名並列時全部顯示', () => {
		const shown = visibleRows(rows([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 13]));
		expect(shown.map((r) => r.rank)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10]);
	});

	it('不足 10 人全數顯示', () => {
		expect(visibleRows(rows([1, 1, 3]))).toHaveLength(3);
	});
});

describe('統計至日期', () => {
	it('ISO 日期轉成「X 月 X 日」', () => {
		expect(formatThroughDate('2026-11-02')).toBe('11 月 2 日');
	});
});

describe('量率散點', () => {
	const men = teams.find((t) => t.group === 'men')!;
	const point = (player_id: number, count: number, rate: number) => ({
		player_id, name: `球員${player_id}`, team_id: men.team_id, team_short: men.short_name, count, rate
	});
	const attack: VolumeRate = {
		key: 'attack_success_rate', median_count: 30, median_rate: 50,
		points: [point(1, 40, 60), point(2, 20, 40)]
	};
	const view = toVolumeRateView(attack, 'men', seasons[0].cup_id);

	it('四種技術依序為攻擊、接發球、防守、舉球，軸名用正式統計名稱', () => {
		expect(VOLUME_RATE_TECHNIQUES.map((t) => t.label)).toEqual(['攻擊', '接發球', '防守', '舉球']);
		expect([view.technique, view.countLabel, view.rateLabel]).toEqual(['攻擊', '攻擊次數', '攻擊成功率']);
		expect(view.points[0].team).toBe(seasonTeam(men, seasons[0].cup_id).short_name);
	});

	it('隊伍對不到 teams.json 時丟錯', () => {
		const broken = { ...attack, points: [{ ...point(1, 40, 60), team_id: -1 }] };
		expect(() => toVolumeRateView(broken, 'men', seasons[0].cup_id)).toThrow('-1');
	});

	it('散點以次數為橫軸、率為縱軸，兩軸中位數畫虛線', () => {
		const option = volumeRateOption(view) as any;
		expect(option.xAxis.name).toBe('攻擊次數');
		expect(option.yAxis.name).toBe('攻擊成功率（%）');
		const [series] = option.series;
		expect(series.type).toBe('scatter');
		expect(series.data.map((d: { value: number[] }) => d.value)).toEqual([[40, 60], [20, 40]]);
		expect(series.markLine.data).toEqual([{ xAxis: 30 }, { yAxis: 50 }]);
		expect(series.markLine.lineStyle.type).toBe('dashed');
	});

	it('點的提示顯示球員、隊名、次數與率', () => {
		const option = volumeRateOption(view) as any;
		const tip = option.tooltip.formatter({ data: option.series[0].data[0] });
		expect(tip).toBe(`球員1（${view.points[0].team}）<br/>攻擊次數：40<br/>攻擊成功率：60%`);
	});

	it('沒有球員達門檻時不產生圖', () => {
		expect(volumeRateOption({ ...view, points: [], median_count: null, median_rate: null })).toBeNull();
	});

	it('說明寫出門檻與兩軸中位數', () => {
		expect(volumeRateNote(view, '門檻：出賽 5 局、攻擊 10 次')).toBe(
			'只列達比率榜門檻（出賽 5 局、攻擊 10 次）的球員；虛線是兩軸中位數（攻擊次數 30、攻擊成功率 50%），右上是量大又準、左下是量少又不準。'
		);
	});
});
