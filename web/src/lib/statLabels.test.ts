import { describe, expect, it } from 'vitest';
import { COUNT_STAT_KEYS, RATE_STAT_KEYS, STAT_LABELS } from './statLabels.js';

describe('統計項目標籤表', () => {
	it('計數榜七項依詞彙表正式名稱', () => {
		expect(COUNT_STAT_KEYS.map((k) => STAT_LABELS[k])).toEqual([
			'總得分',
			'攻擊得分',
			'攔網得分',
			'發球得分',
			'接發球到位',
			'防守到位',
			'舉球到位'
		]);
	});

	it('比率榜七項依詞彙表正式名稱', () => {
		expect(RATE_STAT_KEYS.map((k) => STAT_LABELS[k])).toEqual([
			'攻擊成功率',
			'發球得分率',
			'接發球到位率',
			'防守到位率',
			'舉球到位率',
			'場均得分',
			'每局攔網'
		]);
	});

	it('表裡只有這 14 個項目，鍵名對應匯出資料欄位', () => {
		expect(Object.keys(STAT_LABELS).sort()).toEqual(
			[
				'total_points',
				'attack_points',
				'block_points',
				'serve_points',
				'receive_excellent',
				'dig_excellent',
				'set_excellent',
				'attack_success_rate',
				'serve_point_rate',
				'receive_excellent_rate',
				'dig_excellent_rate',
				'set_excellent_rate',
				'points_per_game',
				'blocks_per_set'
			].sort()
		);
	});
});
