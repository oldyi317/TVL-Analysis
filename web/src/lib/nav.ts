import type { ComponentProps } from 'svelte';
import type Icon from '#lib/components/Icon.svelte';

export type MoreSection = 'players' | 'leaders' | 'records';
export type Section = 'home' | 'matches' | 'standings' | 'teams' | MoreSection;
export type Tab = Exclude<Section, MoreSection> | 'more';

export const NAV_ITEMS: { section: Section; label: string; icon: ComponentProps<typeof Icon>['name'] }[] = [
	{ section: 'home', label: '首頁', icon: 'home' },
	{ section: 'matches', label: '賽事', icon: 'calendar' },
	{ section: 'standings', label: '戰績', icon: 'trophy' },
	{ section: 'teams', label: '球隊', icon: 'shield' },
	{ section: 'players', label: '球員', icon: 'user' },
	{ section: 'leaders', label: '排行榜', icon: 'podium' },
	{ section: 'records', label: '歷史紀錄', icon: 'trophy' }
];

export const MORE_SECTIONS: readonly Section[] = ['players', 'leaders', 'records'];

function isMore(section: Section): section is MoreSection {
	return MORE_SECTIONS.includes(section);
}

export function sectionOf(routeId: string | null): Section | null {
	if (!routeId) return null;
	if (routeId.startsWith('/records/')) return 'records';
	const match = routeId.match(/^\/\[season\]\/\[group\](?:\/([^/]+))?/);
	if (!match) return null;
	const segment = match[1];
	if (segment === undefined) return 'home';
	return NAV_ITEMS.find((i) => i.section === segment && i.section !== 'home' && i.section !== 'records')?.section ?? null;
}

export function tabOf(routeId: string | null): Tab | null {
	const section = sectionOf(routeId);
	if (!section) return null;
	return isMore(section) ? 'more' : section;
}

export function hrefOf(section: Section, base: string, recordsBase: string): string {
	if (section === 'records') return recordsBase;
	return section === 'home' ? base : `${base}${section}/`;
}

export type ListSection = 'matches' | 'teams' | 'players';

const DETAIL_ROUTES: Record<string, ListSection> = {
	'/[season]/[group]/matches/[game]': 'matches',
	'/[season]/[group]/teams/[team]': 'teams',
	'/[season]/[group]/players/[player]': 'players'
};

export function detailListOf(routeId: string | null): ListSection | null {
	return (routeId && DETAIL_ROUTES[routeId]) || null;
}

// base 是含結尾斜線的 `/<base>/<season>/<group>/`；單場頁帶 date 時落在該月份錨點
export function listHref(section: ListSection, base: string, date?: string): string {
	const href = `${base}${section}/`;
	return section === 'matches' && date ? `${href}#m-${date.slice(0, 7)}` : href;
}

export type NavType = 'enter' | 'popstate' | 'goto' | 'link' | 'form' | 'leave';

export function nextDepth(depth: number, type: NavType, delta = 0): number {
	if (type === 'enter') return 0;
	if (type === 'popstate') return Math.max(0, depth + delta);
	return depth + 1;
}
