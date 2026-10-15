import { currentSeason, seasons, teamsIn } from '#lib/site.js';
import type { StandingsRow } from '#lib/components/StandingsTable.svelte';
import type { StyleMatch } from '#lib/match.js';
import type { Team } from '#lib/data/types.js';

// 樣本要五支男隊；新季剛建、尚無隊伍時退回最近有隊伍的賽季
export const demoSeason =
	[...seasons].reverse().find((s) => teamsIn(s.cup_id, 'men').length >= 5) ?? currentSeason;
const men = teamsIn(demoSeason.cup_id, 'men');
const women2024 = teamsIn(20, 'women');

export const sampleMatches: StyleMatch[] = [
	{
		date: '2025-12-06',
		weekday: '六',
		venue: '臺北體育館',
		home: men[0],
		away: men[1],
		sets: [
			[25, 21],
			[22, 25],
			[25, 18],
			[25, 23]
		]
	},
	{
		date: '2025-12-13',
		weekday: '六',
		venue: '桃園巨蛋',
		home: men[2],
		away: men[3],
		sets: [
			[25, 22],
			[20, 25],
			[25, 23],
			[19, 25],
			[13, 15]
		],
		golden_set: [12, 15]
	},
	{
		date: '2025-12-20',
		weekday: '六',
		venue: '高雄巨蛋',
		home: men[4],
		away: men[0],
		sets: [],
		start_time: '19:00',
		home_win_prob: 0.62
	}
];

function standingsRows(list: Team[]): StandingsRow[] {
	return list.map((team, i) => {
		const wins = Math.max(list.length - 1 - i, 0) * 2 + 1;
		const losses = i * 2 + 1;
		const sets_won = wins * 3 + i;
		const sets_lost = losses * 2 + 1;
		const points_for = sets_won * 25 + 40;
		const points_against = sets_lost * 25 + 90 + i * 20;
		return {
			rank: i + 1,
			team,
			wins,
			losses,
			win_pct: Number((wins / (wins + losses)).toFixed(3)),
			points: wins * 3 - i,
			sets_won,
			sets_lost,
			points_for,
			points_against,
			set_ratio: i === list.length - 1 ? null : Number((sets_won / sets_lost).toFixed(3)),
			point_ratio: Number((points_for / points_against).toFixed(3)),
			set_record: {
				'3-0': Math.max(wins - 2, 0),
				'3-1': 1,
				'3-2': 1,
				'2-3': i % 2,
				'1-3': 1,
				'0-3': Math.max(losses - 2, 0)
			},
			last5: (['W', 'W', 'L', 'W', 'L'] as const).map((r, k) =>
				(i + k) % 3 === 0 ? (r === 'W' ? 'L' : 'W') : r
			)
		};
	});
}

export const menStandings = standingsRows(men);
export const womenStandings2024 = standingsRows(women2024);

export const samplePlayers = [
	{ name: '王小明', number: 7, position: '主攻手', team: men[0] },
	{ name: '陳大華', number: null, position: '舉球員', team: men[1] },
	{ name: '林小美', number: 12, position: null, team: women2024[0] }
];

export const trendOption = {
	tooltip: { trigger: 'axis' },
	xAxis: { type: 'category', data: ['1', '2', '3', '4', '5', '6', '7', '8'] },
	yAxis: { type: 'value' },
	series: [{ name: '單場得分', type: 'line', data: [78, 92, 85, 101, 88, 96, 104, 99] }]
};
