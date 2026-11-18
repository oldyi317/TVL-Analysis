import { describe, expect, it } from 'vitest';
import { resultsVersions, todayVersions } from '../src/text';
import type { FeedMatch } from '../src/types';

const ROOT = 'https://vstat.tw';
const LINKS = { men: '/2026-27/men/matches/', women: '/2026-27/women/matches/', both: '/matches/' };
const MEN_URL = 'https://vstat.tw/2026-27/men/matches/';
const WOMEN_URL = 'https://vstat.tw/2026-27/women/matches/';
const BOTH_URL = 'https://vstat.tw/matches/';

function game(overrides: Partial<FeedMatch>): FeedMatch {
	return {
		group: 'men',
		game_id: 1,
		date: '2026-11-08',
		start_time: '14:00',
		home: '臺電',
		away: '雲林美津濃',
		home_sets_won: null,
		away_sets_won: null,
		home_win_prob: null,
		...overrides
	};
}

describe('resultsVersions 三個版本', () => {
	const a = game({ group: 'men', game_id: 1, date: '2026-11-07', start_time: '14:00', home: '臺電', away: '雲林美津濃', home_sets_won: 3, away_sets_won: 1 });
	const b = game({ group: 'men', game_id: 2, date: '2026-11-08', start_time: '12:00', home: '三福產業', away: '臺北鳳凰', home_sets_won: 0, away_sets_won: 3 });
	const c = game({ group: 'women', game_id: 3, date: '2026-11-08', start_time: '12:00', home: '臺中連莊', away: '桃園台灣產險', home_sets_won: 3, away_sets_won: 2 });
	const versions = resultsVersions([c, b, a], LINKS, ROOT);

	it('男子組版', () => {
		expect(versions.men).toEqual({
			title: '男子組賽果',
			body: '11/7 臺電 3–1 雲林美津濃\n11/8 三福產業 0–3 臺北鳳凰',
			navigate: `${MEN_URL}#d-2026-11-08`
		});
	});

	it('女子組版', () => {
		expect(versions.women).toEqual({
			title: '女子組賽果',
			body: '11/8 臺中連莊 3–2 桃園台灣產險',
			navigate: `${WOMEN_URL}#d-2026-11-08`
		});
	});

	it('兩組合併版：標題、每行加組別、連結用 links.both', () => {
		expect(versions.both).toEqual({
			title: '最新賽果',
			body: '11/7 男子 臺電 3–1 雲林美津濃\n11/8 男子 三福產業 0–3 臺北鳳凰\n11/8 女子 臺中連莊 3–2 桃園台灣產險',
			navigate: `${BOTH_URL}#d-2026-11-08`
		});
	});

	it('男子組版與女子組版最新比賽日不同時，各用自己內容裡最新的比賽日', () => {
		const olderWomen = { ...c, date: '2026-11-06' };
		const mixed = resultsVersions([a, b, olderWomen], LINKS, ROOT);
		expect(mixed.men?.navigate).toBe(`${MEN_URL}#d-2026-11-08`);
		expect(mixed.women?.navigate).toBe(`${WOMEN_URL}#d-2026-11-06`);
		expect(mixed.both?.navigate).toBe(`${BOTH_URL}#d-2026-11-08`);
	});

	it('最新比賽日不是輸入順序最後一筆', () => {
		const result = resultsVersions([b, a], LINKS, ROOT);
		expect(result.men?.navigate).toBe(`${MEN_URL}#d-2026-11-08`);
	});

	it('只有男子組時，both 版與男子組版完全相同，女子組版是 null', () => {
		const only = resultsVersions([a, b], LINKS, ROOT);
		expect(only.men).not.toBeNull();
		expect(only.both).toEqual(only.men);
		expect(only.women).toBeNull();
		expect(only.both).toEqual({
			title: '男子組賽果',
			body: '11/7 臺電 3–1 雲林美津濃\n11/8 三福產業 0–3 臺北鳳凰',
			navigate: `${MEN_URL}#d-2026-11-08`
		});
	});

	it('只有女子組時，both 版與女子組版完全相同，男子組版是 null', () => {
		const only = resultsVersions([c], LINKS, ROOT);
		expect(only.women).not.toBeNull();
		expect(only.both).toEqual(only.women);
		expect(only.men).toBeNull();
		expect(only.both?.navigate).toBe(`${WOMEN_URL}#d-2026-11-08`);
	});

	it('沒有賽事時三個都是 null', () => {
		expect(resultsVersions([], LINKS, ROOT)).toEqual({ men: null, women: null, both: null });
	});

	it('siteRoot 帶子路徑時連結正確', () => {
		const preview = resultsVersions([a], LINKS, 'https://tommyyi0317.github.io/tvl-shell-proto');
		expect(preview.men?.navigate).toBe('https://tommyyi0317.github.io/tvl-shell-proto/2026-27/men/matches/#d-2026-11-07');
	});
});

describe('resultsVersions 排序與格式', () => {
	it('同一天先依開賽時間，空值排最後', () => {
		const late = game({ game_id: 1, start_time: null, home: '甲隊', away: '客隊', home_sets_won: 3, away_sets_won: 0 });
		const early = game({ game_id: 2, start_time: '10:00', home: '乙隊', away: '客隊', home_sets_won: 3, away_sets_won: 0 });
		const noon = game({ game_id: 3, start_time: '12:00', home: '丙隊', away: '客隊', home_sets_won: 3, away_sets_won: 0 });
		expect(resultsVersions([late, noon, early], LINKS, ROOT).men?.body).toBe('11/8 乙隊 3–0 客隊\n11/8 丙隊 3–0 客隊\n11/8 甲隊 3–0 客隊');
	});

	it('比賽日優先於開賽時間', () => {
		const earlyDay = game({ game_id: 1, date: '2026-11-07', start_time: '20:00', home: '甲隊', away: '客隊', home_sets_won: 3, away_sets_won: 0 });
		const laterDay = game({ game_id: 2, date: '2026-11-08', start_time: '09:00', home: '乙隊', away: '客隊', home_sets_won: 3, away_sets_won: 0 });
		expect(resultsVersions([laterDay, earlyDay], LINKS, ROOT).men?.body).toBe('11/7 甲隊 3–0 客隊\n11/8 乙隊 3–0 客隊');
	});

	it('同日同時間男子組在前，即使女子組的 game_id 較小', () => {
		const women = game({ group: 'women', game_id: 1, home: '女甲', away: '客隊', home_sets_won: 3, away_sets_won: 0 });
		const men = game({ group: 'men', game_id: 9, home: '男甲', away: '客隊', home_sets_won: 3, away_sets_won: 0 });
		expect(resultsVersions([women, men], LINKS, ROOT).both?.body).toBe('11/8 男子 男甲 3–0 客隊\n11/8 女子 女甲 3–0 客隊');
	});

	it('同組同時間依 game_id 由小到大', () => {
		const second = game({ game_id: 5, home: '乙隊', away: '客隊', home_sets_won: 3, away_sets_won: 0 });
		const first = game({ game_id: 4, home: '甲隊', away: '客隊', home_sets_won: 3, away_sets_won: 0 });
		expect(resultsVersions([second, first], LINKS, ROOT).men?.body).toBe('11/8 甲隊 3–0 客隊\n11/8 乙隊 3–0 客隊');
	});

	it('日期的月與日不補零', () => {
		const single = game({ date: '2026-01-05', home: '甲隊', away: '客隊', home_sets_won: 3, away_sets_won: 2 });
		expect(resultsVersions([single], LINKS, ROOT).men?.body).toBe('1/5 甲隊 3–2 客隊');
	});

	it('比分的分隔是 en dash U+2013', () => {
		const one = game({ home_sets_won: 3, away_sets_won: 1 });
		expect(resultsVersions([one], LINKS, ROOT).men?.body).toContain('–');
		expect(resultsVersions([one], LINKS, ROOT).men?.body).not.toContain('3-1');
	});
});

describe('resultsVersions 8 行上限', () => {
	const names = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬'];
	function nine(count: number) {
		return names.slice(0, count).map((name, index) => game({ game_id: index + 1, date: `2026-11-0${index + 1}`, start_time: '14:00', home: `${name}隊`, away: '客隊', home_sets_won: 3, away_sets_won: 0 }));
	}

	it('剛好 8 場列 8 行', () => {
		expect(resultsVersions(nine(8), LINKS, ROOT).men?.body).toBe(
			['11/1 甲隊 3–0 客隊', '11/2 乙隊 3–0 客隊', '11/3 丙隊 3–0 客隊', '11/4 丁隊 3–0 客隊', '11/5 戊隊 3–0 客隊', '11/6 己隊 3–0 客隊', '11/7 庚隊 3–0 客隊', '11/8 辛隊 3–0 客隊'].join('\n')
		);
	});

	it('9 場列 7 行加「另有 2 場」', () => {
		expect(resultsVersions(nine(9), LINKS, ROOT).men?.body).toBe(
			['11/1 甲隊 3–0 客隊', '11/2 乙隊 3–0 客隊', '11/3 丙隊 3–0 客隊', '11/4 丁隊 3–0 客隊', '11/5 戊隊 3–0 客隊', '11/6 己隊 3–0 客隊', '11/7 庚隊 3–0 客隊', '另有 2 場'].join('\n')
		);
	});

	it('連結仍用被截掉那幾場在內的最新比賽日', () => {
		expect(resultsVersions(nine(9), LINKS, ROOT).men?.navigate).toBe(`${MEN_URL}#d-2026-11-09`);
	});

	it('兩組合併後總數超過 8 場，總場數算進「另有」', () => {
		const mixed = [...nine(5), ...nine(5).map((m) => ({ ...m, group: 'women' as const, game_id: m.game_id + 10 }))];
		const lines = resultsVersions(mixed, LINKS, ROOT).both?.body.split('\n');
		expect(lines).toHaveLength(8);
		expect(lines?.[7]).toBe('另有 3 場');
	});
});

describe('todayVersions 三個版本', () => {
	const t1 = game({ group: 'men', game_id: 10, start_time: '14:00', home: '臺電', away: '雲林美津濃', home_win_prob: 0.62 });
	const t2 = game({ group: 'men', game_id: 11, start_time: '12:00', home: '雲林美津濃', away: '三福產業', home_win_prob: null });
	const t3 = game({ group: 'women', game_id: 12, start_time: '14:00', home: '臺中連莊', away: '桃園台灣產險', home_win_prob: 0.3762 });
	const t4 = game({ group: 'men', game_id: 13, start_time: null, home: '龍邁建設', away: '臺北鳳凰', home_win_prob: null });
	const versions = todayVersions([t4, t3, t1, t2], LINKS, ROOT, '2026-11-08');

	it('男子組版', () => {
		expect(versions.men).toEqual({
			title: '今日男子組賽事',
			body: '12:00 雲林美津濃 對 三福產業\n14:00 臺電 62%–38% 雲林美津濃\n龍邁建設 對 臺北鳳凰',
			navigate: `${MEN_URL}#d-2026-11-08`
		});
	});

	it('女子組版', () => {
		expect(versions.women).toEqual({
			title: '今日女子組賽事',
			body: '14:00 臺中連莊 38%–62% 桃園台灣產險',
			navigate: `${WOMEN_URL}#d-2026-11-08`
		});
	});

	it('兩組合併版：時間後加組別，沒有時間的行以組別開頭', () => {
		expect(versions.both).toEqual({
			title: '今日賽事',
			body: '12:00 男子 雲林美津濃 對 三福產業\n14:00 男子 臺電 62%–38% 雲林美津濃\n14:00 女子 臺中連莊 38%–62% 桃園台灣產險\n男子 龍邁建設 對 臺北鳳凰',
			navigate: `${BOTH_URL}#d-2026-11-08`
		});
	});

	it('連結用傳入的日期，不是賽事上的日期', () => {
		const other = todayVersions([t1], LINKS, ROOT, '2026-12-25');
		expect(other.men?.navigate).toBe(`${MEN_URL}#d-2026-12-25`);
	});

	it('只有女子組時，both 版與女子組版完全相同，男子組版是 null', () => {
		const only = todayVersions([t3], LINKS, ROOT, '2026-11-08');
		expect(only.women).not.toBeNull();
		expect(only.both).toEqual(only.women);
		expect(only.men).toBeNull();
		expect(only.both?.navigate).toBe(`${WOMEN_URL}#d-2026-11-08`);
	});

	it('只有男子組時，both 版與男子組版完全相同，女子組版是 null', () => {
		const only = todayVersions([t1, t2], LINKS, ROOT, '2026-11-08');
		expect(only.men).not.toBeNull();
		expect(only.both).toEqual(only.men);
		expect(only.women).toBeNull();
		expect(only.both?.title).toBe('今日男子組賽事');
	});

	it('沒有賽事時三個都是 null', () => {
		expect(todayVersions([], LINKS, ROOT, '2026-11-08')).toEqual({ men: null, women: null, both: null });
	});

	it('siteRoot 帶子路徑時連結正確', () => {
		const preview = todayVersions([t3], LINKS, 'https://tommyyi0317.github.io/tvl-shell-proto', '2026-11-08');
		expect(preview.women?.navigate).toBe('https://tommyyi0317.github.io/tvl-shell-proto/2026-27/women/matches/#d-2026-11-08');
	});
});

describe('todayVersions 勝率與排序', () => {
	it('勝率 0.376… 顯示 38%–62%', () => {
		const one = game({ home_win_prob: 0.3762 });
		expect(todayVersions([one], LINKS, ROOT, '2026-11-08').men?.body).toBe('14:00 臺電 38%–62% 雲林美津濃');
	});

	it('勝率 0.625 顯示 63%–37%（主隊四捨五入、客隊用 100 減）', () => {
		const one = game({ home_win_prob: 0.625 });
		expect(todayVersions([one], LINKS, ROOT, '2026-11-08').men?.body).toBe('14:00 臺電 63%–37% 雲林美津濃');
	});

	it('勝率 0 與 1 照樣顯示百分比，不當成沒有勝率', () => {
		const zero = game({ game_id: 1, start_time: '10:00', home_win_prob: 0 });
		const one = game({ game_id: 2, start_time: '11:00', home_win_prob: 1 });
		expect(todayVersions([zero, one], LINKS, ROOT, '2026-11-08').men?.body).toBe('10:00 臺電 0%–100% 雲林美津濃\n11:00 臺電 100%–0% 雲林美津濃');
	});

	it('沒有勝率用「對」', () => {
		const one = game({ home_win_prob: null });
		expect(todayVersions([one], LINKS, ROOT, '2026-11-08').men?.body).toBe('14:00 臺電 對 雲林美津濃');
	});

	it('開賽時間是空值的行不寫時間，有勝率時也是', () => {
		const one = game({ start_time: null, home_win_prob: 0.62 });
		expect(todayVersions([one], LINKS, ROOT, '2026-11-08').men?.body).toBe('臺電 62%–38% 雲林美津濃');
	});

	it('空值排最後，早的時間在前', () => {
		const none = game({ game_id: 1, start_time: null, home: '甲隊', away: '客隊' });
		const late = game({ game_id: 2, start_time: '19:00', home: '乙隊', away: '客隊' });
		const early = game({ game_id: 3, start_time: '09:00', home: '丙隊', away: '客隊' });
		expect(todayVersions([none, late, early], LINKS, ROOT, '2026-11-08').men?.body).toBe('09:00 丙隊 對 客隊\n19:00 乙隊 對 客隊\n甲隊 對 客隊');
	});

	it('同時間男子組在前，即使女子組的 game_id 較小', () => {
		const women = game({ group: 'women', game_id: 1, home: '女甲', away: '客隊' });
		const men = game({ group: 'men', game_id: 9, home: '男甲', away: '客隊' });
		expect(todayVersions([women, men], LINKS, ROOT, '2026-11-08').both?.body).toBe('14:00 男子 男甲 對 客隊\n14:00 女子 女甲 對 客隊');
	});

	it('同組同時間依 game_id 由小到大', () => {
		const second = game({ game_id: 5, home: '乙隊', away: '客隊' });
		const first = game({ game_id: 4, home: '甲隊', away: '客隊' });
		expect(todayVersions([second, first], LINKS, ROOT, '2026-11-08').men?.body).toBe('14:00 甲隊 對 客隊\n14:00 乙隊 對 客隊');
	});

	it('今日賽事不看比賽日排序，只看開賽時間', () => {
		const a = game({ game_id: 1, date: '2026-11-09', start_time: '10:00', home: '甲隊', away: '客隊' });
		const b = game({ game_id: 2, date: '2026-11-08', start_time: '11:00', home: '乙隊', away: '客隊' });
		expect(todayVersions([b, a], LINKS, ROOT, '2026-11-08').men?.body).toBe('10:00 甲隊 對 客隊\n11:00 乙隊 對 客隊');
	});
});

describe('todayVersions 8 行上限', () => {
	const names = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬'];
	function nine(count: number) {
		return names.slice(0, count).map((name, index) => game({ game_id: index + 1, start_time: `0${index + 1}:00`, home: `${name}隊`, away: '客隊' }));
	}

	it('剛好 8 場列 8 行', () => {
		expect(todayVersions(nine(8), LINKS, ROOT, '2026-11-08').men?.body).toBe(
			['01:00 甲隊 對 客隊', '02:00 乙隊 對 客隊', '03:00 丙隊 對 客隊', '04:00 丁隊 對 客隊', '05:00 戊隊 對 客隊', '06:00 己隊 對 客隊', '07:00 庚隊 對 客隊', '08:00 辛隊 對 客隊'].join('\n')
		);
	});

	it('9 場列 7 行加「另有 2 場」', () => {
		expect(todayVersions(nine(9), LINKS, ROOT, '2026-11-08').men?.body).toBe(
			['01:00 甲隊 對 客隊', '02:00 乙隊 對 客隊', '03:00 丙隊 對 客隊', '04:00 丁隊 對 客隊', '05:00 戊隊 對 客隊', '06:00 己隊 對 客隊', '07:00 庚隊 對 客隊', '另有 2 場'].join('\n')
		);
	});
});
