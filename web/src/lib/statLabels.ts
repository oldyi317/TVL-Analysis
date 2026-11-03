// 統計項目正式名稱的唯一來源（CONTEXT.md「排行榜」詞條）
export const STAT_LABELS = {
	total_points: '總得分',
	attack_points: '攻擊得分',
	block_points: '攔網得分',
	serve_points: '發球得分',
	receive_excellent: '接發球到位',
	dig_excellent: '防守到位',
	set_excellent: '舉球到位',
	attack_success_rate: '攻擊成功率',
	serve_point_rate: '發球得分率',
	receive_excellent_rate: '接發球到位率',
	dig_excellent_rate: '防守到位率',
	set_excellent_rate: '舉球到位率',
	points_per_game: '場均得分',
	blocks_per_set: '每局攔網'
} as const;

export type StatKey = keyof typeof STAT_LABELS;

export const COUNT_STAT_KEYS = [
	'total_points',
	'attack_points',
	'block_points',
	'serve_points',
	'receive_excellent',
	'dig_excellent',
	'set_excellent'
] as const satisfies readonly StatKey[];

export const RATE_STAT_KEYS = [
	'attack_success_rate',
	'serve_point_rate',
	'receive_excellent_rate',
	'dig_excellent_rate',
	'set_excellent_rate',
	'points_per_game',
	'blocks_per_set'
] as const satisfies readonly StatKey[];

export type CountStatKey = (typeof COUNT_STAT_KEYS)[number];
export type RateStatKey = (typeof RATE_STAT_KEYS)[number];
