import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import type { PostgameComparison as PostgameComparisonData } from '#lib/teamAdvanced.js';
import PostgameComparison from './PostgameComparison.svelte';

const metric = (value: number | null) => ({ numerator: 0, denominator: value === null ? 0 : 10, value });

const sample: PostgameComparisonData = {
	home: {
		game_id: 10,
		date: '2026-01-10',
		opponent_id: 1,
		points: { attack: 40, block: 6, serve: 5, opponent_error: 24, total: 75 },
		metrics: {
			attack_success_rate: metric(45),
			serve_point_rate: metric(null),
			receive_excellent_rate: metric(50),
			dig_excellent_rate: metric(60),
			set_excellent_rate: metric(48),
			blocks_per_set: metric(2)
		}
	},
	away: {
		game_id: 10,
		date: '2026-01-10',
		opponent_id: 4,
		points: { attack: 35, block: 8, serve: 5, opponent_error: -3, total: 65 },
		metrics: {
			attack_success_rate: metric(40),
			serve_point_rate: metric(10),
			receive_excellent_rate: metric(50),
			dig_excellent_rate: metric(55),
			set_excellent_rate: metric(52),
			blocks_per_set: metric(2.67)
		}
	}
};

describe('PostgameComparison', () => {
	const html = () =>
		render(PostgameComparison, { props: { comparison: sample, homeName: '臺北鳳凰', awayName: '屏東台電' } })
			.body.replace(/<!--[\s\S]*?-->/g, '');

	it('輸出得分來源與球隊指標兩張對照表，主隊在左、客隊在右，具欄列標頭', () => {
		const body = html();
		expect(body).toContain('aria-label="得分來源"');
		expect(body).toContain('aria-label="球隊指標"');
		expect(body).toMatch(/<th scope="col"[^>]*>項目<\/th>\s*<th scope="col"[^>]*>臺北鳳凰<\/th>\s*<th scope="col"[^>]*>屏東台電<\/th>/);
		expect(body).toMatch(/<th scope="col"[^>]*>指標<\/th>\s*<th scope="col"[^>]*>臺北鳳凰<\/th>\s*<th scope="col"[^>]*>屏東台電<\/th>/);
		for (const label of ['攻擊得分', '攔網得分', '發球得分', '對手失誤得分', '總得分']) {
			expect(body).toMatch(new RegExp(`<th scope="row"[^>]*>${label}</th>`));
		}
		for (const label of ['攻擊成功率', '發球得分率', '接發球到位率', '防守到位率', '舉球到位率', '每局攔網']) {
			expect(body).toMatch(new RegExp(`<th scope="row"[^>]*>${label}</th>`));
		}
	});

	it('每列較高的一方以 strong 強調，同值、負值對手失誤得分與分母為 0 皆不標示且顯示「—」', () => {
		const body = html();
		// 攻擊得分 40 > 35（主隊較高）
		expect(body).toMatch(/<th scope="row"[^>]*>攻擊得分<\/th>\s*<td[^>]*class="[^"]*strong[^"]*"[^>]*><strong[^>]*>40<\/strong><\/td>\s*<td[^>]*>35<\/td>/);
		// 攔網得分 6 < 8（客隊較高）
		expect(body).toMatch(/<th scope="row"[^>]*>攔網得分<\/th>\s*<td[^>]*>6<\/td>\s*<td[^>]*class="[^"]*strong[^"]*"[^>]*><strong[^>]*>8<\/strong><\/td>/);
		// 發球得分 5 = 5（同值不標示）
		expect(body).toMatch(/<th scope="row"[^>]*>發球得分<\/th>\s*<td[^>]*>5<\/td>\s*<td[^>]*>5<\/td>/);
		// 對手失誤得分客隊負值顯示「—」且雙方不標示
		expect(body).toMatch(/<th scope="row"[^>]*>對手失誤得分<\/th>\s*<td[^>]*>24<\/td>\s*<td[^>]*>—<\/td>/);
		// 發球得分率主隊 null 顯示「—」且雙方不標示
		expect(body).toMatch(/<th scope="row"[^>]*>發球得分率<\/th>\s*<td[^>]*>—<\/td>\s*<td[^>]*>10%<\/td>/);
		// 每局攔網客隊較高
		expect(body).toMatch(/<th scope="row"[^>]*>每局攔網<\/th>\s*<td[^>]*>2<\/td>\s*<td[^>]*class="[^"]*strong[^"]*"[^>]*><strong[^>]*>2\.67<\/strong><\/td>/);
	});
});
