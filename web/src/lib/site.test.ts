import { describe, expect, it } from 'vitest';
import playersJson from '#lib/data/players.json';
import type {
	MatchEntry,
	PlayerEntry,
	PlayersData,
	RosterWeek,
	StandingsEntry
} from '#lib/data/types.js';
import {
	findMatch,
	findPlayerSeason,
	findSeason,
	findTeam,
	matchesFor,
	pickLatestWeek,
	playerDisplayName,
	positionLabel,
	playerEntries,
	playerName,
	playersByTeam,
	standingsFor,
	teamEntries,
	teamMatches,
	teams,
	toMatchView,
	toRosterView,
	toStandingsRow
} from './site.js';

const entry: StandingsEntry = {
	rank: 1,
	team_id: 3,
	wins: 5,
	losses: 1,
	win_pct: 0.833,
	set_record: { '3-0': 2, '3-1': 2, '3-2': 1, '2-3': 0, '1-3': 1, '0-3': 0 },
	points: 14,
	sets_won: 16,
	sets_lost: 7,
	set_ratio: null,
	points_for: 400,
	points_against: 350,
	point_ratio: 1.143,
	last5: ['W', 'W', 'L', 'W', 'W']
};

describe('toStandingsRow', () => {
	it('以 group + team_id 對回球隊並帶過數值', () => {
		const real = teams.find((t) => t.group === 'women')!;
		const row = toStandingsRow({ ...entry, team_id: real.team_id }, 'women');
		expect(row.team).toBe(real);
		expect(row).toMatchObject({ rank: 1, wins: 5, losses: 1, points: 14, set_ratio: null });
		expect(row).not.toHaveProperty('team_id');
		expect(row.last5).toEqual(['W', 'W', 'L', 'W', 'W']);
	});
	it('帶過勝率、局數與得失分欄位', () => {
		const real = teams.find((t) => t.group === 'men')!;
		const row = toStandingsRow({ ...entry, team_id: real.team_id }, 'men');
		expect(row).toMatchObject({
			win_pct: 0.833,
			sets_won: 16,
			sets_lost: 7,
			points_for: 400,
			points_against: 350
		});
	});
	it('對不到球隊就丟錯', () => {
		expect(() => toStandingsRow({ ...entry, team_id: -1 }, 'men')).toThrow('team_id -1');
	});
});

describe('standingsFor', () => {
	it('找不到的賽季回 undefined', () => {
		expect(standingsFor(-1, 'men')).toBeUndefined();
	});
});

describe('toMatchView', () => {
	const base: MatchEntry = {
		game_id: 327,
		date: '2026-03-22',
		weekday: '日',
		start_time: null,
		venue: '國立臺灣大學綜合體育館',
		stage: '總決賽',
		week: 19,
		home_team_id: 1,
		away_team_id: 2,
		sets: [[25, 20], [25, 18], [25, 22]],
		home_sets_won: 3,
		away_sets_won: 0,
		golden_set: [13, 15],
		home_win_prob: null
	};
	it('帶過主隊賽前勝率', () => {
		expect(toMatchView({ ...base, home_win_prob: 0.62 }, 'men').home_win_prob).toBe(0.62);
	});
	it('以 group + team_id 對回兩隊並帶過比分、階段與黃金決勝局', () => {
		const men = teams.filter((t) => t.group === 'men');
		const view = toMatchView({ ...base, home_team_id: men[0].team_id, away_team_id: men[1].team_id }, 'men');
		expect(view.home).toBe(men[0]);
		expect(view.away).toBe(men[1]);
		expect(view).toMatchObject({
			game_id: 327,
			date: '2026-03-22',
			weekday: '日',
			venue: '國立臺灣大學綜合體育館',
			stage: '總決賽',
			week: 19,
			golden_set: [13, 15],
			start_time: null
		});
		expect(view.sets).toEqual([[25, 20], [25, 18], [25, 22]]);
	});
	it('場館缺值顯示「場館未定」', () => {
		const men = teams.filter((t) => t.group === 'men');
		const view = toMatchView(
			{ ...base, venue: null, home_team_id: men[0].team_id, away_team_id: men[1].team_id },
			'men'
		);
		expect(view.venue).toBe('場館未定');
	});
	it('對不到球隊就丟錯', () => {
		expect(() => toMatchView({ ...base, home_team_id: -1 }, 'men')).toThrow('team_id -1');
	});
});

describe('matchesFor / findMatch', () => {
	it('找不到的賽季回空陣列與 undefined', () => {
		expect(matchesFor(-1, 'men')).toEqual([]);
		expect(findMatch(-1, 'men', 1)).toBeUndefined();
	});
});

describe('findTeam', () => {
	const inactive = teams.find((t) => !t.is_active)!;
	const active = teams.find((t) => t.is_active)!;
	it('非現役隊回 undefined', () => {
		expect(findTeam(inactive.cup_ids[0], inactive.group, inactive.team_id)).toBeUndefined();
	});
	it('該季不在 cup_ids 回 undefined', () => {
		expect(findTeam(-1, active.group, active.team_id)).toBeUndefined();
	});
	it('現役且該季有的隊會回傳', () => {
		expect(findTeam(active.cup_ids[0], active.group, active.team_id)).toBe(active);
	});
});

describe('teamEntries', () => {
	it('只含現役且該季有的隊，不含非現役隊', () => {
		const entries = teamEntries();
		expect(entries.length).toBeGreaterThan(0);
		for (const e of entries) {
			expect(findTeam(findSeason(e.season)!.cup_id, e.group, Number(e.team))).toBeDefined();
		}
		const inactive = teams.find((t) => !t.is_active)!;
		expect(
			entries.some((e) => e.group === inactive.group && e.team === String(inactive.team_id))
		).toBe(false);
	});
});

describe('teamMatches', () => {
	it('只含該隊的賽事並依日期、開賽時間排序', () => {
		const season = findSeason('2025-26')!;
		const teamId = matchesFor(season.cup_id, 'men')[0].home.team_id;
		const list = teamMatches(season.cup_id, 'men', teamId);
		expect(list.length).toBeGreaterThan(0);
		for (const m of list) expect([m.home.team_id, m.away.team_id]).toContain(teamId);
		const keys = list.map((m) => `${m.date} ${m.start_time ?? ''}`);
		expect(keys).toEqual([...keys].sort());
	});
});

describe('pickLatestWeek', () => {
	const week = (label: string, date: string | null): RosterWeek => ({
		week_label: label,
		week_start_date: date,
		players: []
	});
	it('取日期最大的週，null 視為最舊', () => {
		const picked = pickLatestWeek([
			week('B', '2026-03-01'),
			week('N', null),
			week('C', '2026-03-22'),
			week('A', '2026-02-01')
		]);
		expect(picked?.week_label).toBe('C');
	});
	it('只有 null 日期時仍回傳第一筆', () => {
		expect(pickLatestWeek([week('N', null)])?.week_label).toBe('N');
	});
	it('空陣列回 undefined', () => {
		expect(pickLatestWeek([])).toBeUndefined();
	});
});

describe('playerName', () => {
	const entries = [{ player_id: 1, group: 'men', name: '王小明' }, { player_id: 2, group: 'men', name: null }] as PlayerEntry[];

	it('依 group 與 player_id 找姓名，找不到或姓名為空用 fallback', () => {
		expect(playerName('men', 1, entries)).toBe('王小明');
		expect(playerName('women', 1, entries)).toBe('球員 #1');
		expect(playerName('men', 2, entries)).toBe('球員 #2');
	});
});

describe('toRosterView', () => {
	const entries = [
		{ player_id: 1, group: 'men', name: '王小明' },
		{ player_id: 2, group: 'men', name: null },
		{ player_id: 3, group: 'women', name: '女生三' }
	] as PlayerEntry[];
	const week: RosterWeek = {
		week_label: 'Week 1',
		week_start_date: '2026-01-01',
		players: [
			{ player_id: 1, jersey_number: 7, position: 'S', source: 'match_page' },
			{ player_id: 2, jersey_number: null, position: null, source: 'backfill' },
			{ player_id: 3, jersey_number: 5, position: 'L', source: 'backfill' }
		]
	};
	it('以 (player_id, group) 對姓名，對不到或姓名為 null 用備援名稱', () => {
		const view = toRosterView(week, 'men', entries);
		expect(view.week_label).toBe('Week 1');
		expect(view.players.map((p) => p.name)).toEqual(['王小明', '球員 #2', '球員 #3']);
		expect(view.players[1]).toMatchObject({ number: null, position: null });
	});
});

const realPlayers = (playersJson as PlayersData).players;
const CUP = 21;

describe('playerEntries', () => {
	it('每筆都對得到該季該組別的登錄，數量等於所有賽季登錄數', () => {
		const entries = playerEntries();
		for (const e of entries) {
			const cupId = findSeason(e.season)!.cup_id;
			expect(findPlayerSeason(cupId, e.group, Number(e.player))).toBeDefined();
		}
		expect(entries).toHaveLength(realPlayers.reduce((n, p) => n + p.seasons.length, 0));
	});
});

describe('playersByTeam', () => {
	const groups = playersByTeam(CUP, 'men');
	it('球隊順序照 teams.json，隊內背號升冪且 null 排最後', () => {
		expect(groups.length).toBeGreaterThan(0);
		const order = groups.map((g) => teams.indexOf(g.team));
		expect(order).toEqual([...order].sort((a, b) => a - b));
		for (const g of groups) {
			const nums = g.players.map((p) => p.number ?? Infinity);
			expect(nums).toEqual([...nums].sort((a, b) => a - b));
		}
	});
	it('球員總數等於該季該組別的登錄數', () => {
		const expected = realPlayers.filter(
			(p) => p.group === 'men' && p.seasons.some((s) => s.cup_id === CUP)
		).length;
		expect(groups.flatMap((g) => g.players)).toHaveLength(expected);
	});
	it('沒有登錄的賽季回空陣列', () => {
		expect(playersByTeam(20, 'men')).toEqual([]);
	});
});

describe('findPlayerSeason', () => {
	const player = realPlayers.find((p) => p.seasons.some((s) => s.cup_id === CUP))!;
	it('存在時回球員、該季資料與球隊', () => {
		const found = findPlayerSeason(CUP, player.group, player.player_id);
		expect(found?.player).toBe(player);
		expect(found?.season.cup_id).toBe(CUP);
		expect(found?.team.team_id).toBe(found?.season.team_id);
	});
	it('該季沒有登錄回 undefined', () => {
		expect(findPlayerSeason(20, player.group, player.player_id)).toBeUndefined();
	});
	it('同一個 player_id 換組別不會拿到原本那位', () => {
		const otherGroup = player.group === 'men' ? 'women' : 'men';
		const other = findPlayerSeason(CUP, otherGroup, player.player_id);
		expect(other === undefined || (other.player !== player && other.player.group === otherGroup)).toBe(
			true
		);
	});
	it('沒有這位球員回 undefined', () => {
		expect(findPlayerSeason(CUP, player.group, -1)).toBeUndefined();
	});
});

describe('playerDisplayName', () => {
	it('有姓名用姓名，null 用備援名稱', () => {
		expect(playerDisplayName({ player_id: 1, name: '王小明' })).toBe('王小明');
		expect(playerDisplayName({ player_id: 2, name: null })).toBe('球員 #2');
	});
});

describe('positionLabel', () => {
	it('位置代碼轉兩字簡稱，可附代碼', () => {
		expect(positionLabel('MB')).toBe('攔中');
		expect(positionLabel('OP', true)).toBe('副攻（OP）');
		expect(positionLabel('S')).toBe('舉球');
		expect(positionLabel('L', true)).toBe('自由（L）');
		expect(positionLabel('OH')).toBe('主攻');
	});

	it('沒有位置給「—」，未知代碼照原樣', () => {
		expect(positionLabel(null)).toBe('—');
		expect(positionLabel('XX', true)).toBe('XX');
	});
});
