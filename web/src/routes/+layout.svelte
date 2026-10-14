<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { asset } from '$app/paths';
	import SiteHeader from '#lib/components/SiteHeader.svelte';
	import { browserStorage, writeGroup } from '#lib/groupMemory.js';
	import { findSeason, findTeam, isGroup } from '#lib/site.js';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();

	// 切換賽季或組別：只替換路徑中該片段的第一次出現，停留在同一頁面類型。
	// base 是固定的 /TVL-Analysis，賽季與組別是 base 之後的前兩段，不會與其他片段撞名。
	// 單場賽事只屬於一個賽季與組別，從詳情頁切換就退回該組別的賽事列表。
	// 球隊頁同理：切組別退回球隊列表，切賽季只在目標賽季也有這支球隊時保留。
	function switchHref(param: 'season' | 'group', value: string): string | undefined {
		const current = page.params[param];
		if (!current) return undefined;
		const path = page.params.game
			? page.url.pathname.replace(`/matches/${page.params.game}/`, '/matches/')
			: page.url.pathname;
		const team = page.params.team;
		const switched = (p: string) => p.replace(`/${current}/`, `/${value}/`);
		if (!team) return switched(path);
		if (keepTeamPage(param, value, team)) return switched(path);
		return switched(path.replace(`/teams/${team}/`, '/teams/'));
	}

	// team_id 跨組別是不同隊；跨賽季則看目標賽季有沒有這支現役隊
	function keepTeamPage(param: 'season' | 'group', value: string, team: string): boolean {
		if (param !== 'season') return false;
		const group = page.params.group;
		const cupId = findSeason(value)?.cup_id;
		if (!group || !isGroup(group) || cupId === undefined) return false;
		return findTeam(cupId, group, Number(team)) !== undefined;
	}

	const group = $derived(page.params.group && isGroup(page.params.group) ? page.params.group : undefined);

	$effect(() => {
		if (group) writeGroup(browserStorage(), group);
	});
</script>

<svelte:head>
	<link rel="icon" href={asset('favicon.svg')} />
</svelte:head>

<SiteHeader season={page.params.season} {group} hrefFor={switchHref} />

<main class="wrap">
	{@render children()}
</main>

<style>
	.wrap {
		max-inline-size: var(--measure);
		margin-inline: auto;
		padding-inline: var(--space-4);
	}

	main {
		padding-block-end: var(--space-7);
	}
</style>
