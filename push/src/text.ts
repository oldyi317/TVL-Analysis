import type { Content, FeedLinks, FeedMatch, Group, GroupPref, Versions } from './types';

const MAX_LINES = 8;
const GROUP_LABEL: Record<Group, string> = { men: '男子', women: '女子' };

interface Spec {
	titles: Record<GroupPref, string>;
	compare: (a: FeedMatch, b: FeedMatch) => number;
	line: (match: FeedMatch, tagged: boolean) => string;
	anchorDate: (matches: FeedMatch[]) => string;
}

function compareTime(a: FeedMatch, b: FeedMatch): number {
	if (a.start_time === b.start_time) return 0;
	if (a.start_time === null) return 1;
	if (b.start_time === null) return -1;
	return a.start_time < b.start_time ? -1 : 1;
}

function compareGroupThenId(a: FeedMatch, b: FeedMatch): number {
	if (a.group !== b.group) return a.group === 'men' ? -1 : 1;
	return a.game_id - b.game_id;
}

function compareDate(a: FeedMatch, b: FeedMatch): number {
	if (a.date === b.date) return 0;
	return a.date < b.date ? -1 : 1;
}

// 月/日不補零。
function shortDate(date: string): string {
	const [, month, day] = date.split('-');
	return `${Number(month)}/${Number(day)}`;
}

function percent(match: FeedMatch): string | null {
	if (match.home_win_prob === null) return null;
	const home = Math.round(match.home_win_prob * 100);
	return `${home}%–${100 - home}%`;
}

function content(matches: FeedMatch[], pref: GroupPref, links: FeedLinks, siteRoot: string, spec: Spec): Content | null {
	if (matches.length === 0) return null;
	const sorted = [...matches].sort(spec.compare);
	const tagged = pref === 'both';
	const shown = sorted.length > MAX_LINES ? sorted.slice(0, MAX_LINES - 1) : sorted;
	const lines = shown.map((match) => spec.line(match, tagged));
	if (shown.length < sorted.length) lines.push(`另有 ${sorted.length - shown.length} 場`);
	return {
		title: spec.titles[pref],
		body: lines.join('\n'),
		navigate: `${siteRoot}${links[pref]}#d-${spec.anchorDate(matches)}`
	};
}

function versions(matches: FeedMatch[], links: FeedLinks, siteRoot: string, spec: Spec): Versions {
	const men = content(matches.filter((match) => match.group === 'men'), 'men', links, siteRoot, spec);
	const women = content(matches.filter((match) => match.group === 'women'), 'women', links, siteRoot, spec);
	// 只有一組時 both 版就是那一組的版本（標題、內文、連結都一樣）。
	const both = men && women ? content(matches, 'both', links, siteRoot, spec) : (men ?? women);
	return { men, women, both };
}

export function resultsVersions(matches: FeedMatch[], links: FeedLinks, siteRoot: string): Versions {
	return versions(matches, links, siteRoot, {
		titles: { men: '男子組賽果', women: '女子組賽果', both: '最新賽果' },
		compare: (a, b) => compareDate(a, b) || compareTime(a, b) || compareGroupThenId(a, b),
		line: (match, tagged) => {
			const tag = tagged ? ` ${GROUP_LABEL[match.group]}` : '';
			return `${shortDate(match.date)}${tag} ${match.home} ${match.home_sets_won}–${match.away_sets_won} ${match.away}`;
		},
		anchorDate: (list) => list.reduce((latest, match) => (match.date > latest ? match.date : latest), list[0].date)
	});
}

export function todayVersions(matches: FeedMatch[], links: FeedLinks, siteRoot: string, date: string): Versions {
	return versions(matches, links, siteRoot, {
		titles: { men: '今日男子組賽事', women: '今日女子組賽事', both: '今日賽事' },
		compare: (a, b) => compareTime(a, b) || compareGroupThenId(a, b),
		line: (match, tagged) => {
			const parts = [match.start_time, tagged ? GROUP_LABEL[match.group] : null, match.home, percent(match) ?? '對', match.away];
			return parts.filter((part) => part !== null).join(' ');
		},
		anchorDate: () => date
	});
}
