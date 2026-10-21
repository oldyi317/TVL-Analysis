<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { asset } from '$app/paths';
	import SiteFooter from '#lib/components/SiteFooter.svelte';
	import SiteHeader from '#lib/components/SiteHeader.svelte';
	import { browserStorage, writeGroup } from '#lib/groupMemory.js';
	import { findPlayerSeason, findSeason, findTeam, isGroup } from '#lib/site.js';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();

	// 切換賽季或組別：只替換路徑中該片段的第一次出現，停留在同一頁面類型。
	// base 是固定的 /TVL-Analysis，賽季與組別是 base 之後的前兩段，不會與其他片段撞名。
	// 單場賽事只屬於一個賽季與組別，從詳情頁切換就退回該組別的賽事列表。
	// 球隊頁、球員頁同理：切組別退回列表，切賽季只在目標賽季也有這支球隊／這位球員時保留。
	function switchHref(param: 'season' | 'group', value: string): string | undefined {
		const current = page.params[param];
		if (!current) return undefined;
		const path = page.params.game
			? page.url.pathname.replace(`/matches/${page.params.game}/`, '/matches/')
			: page.url.pathname;
		const team = page.params.team;
		const player = page.params.player;
		const switched = (p: string) => p.replace(`/${current}/`, `/${value}/`);
		if (team) {
			if (keepTeamPage(param, value, team)) return switched(path);
			return switched(path.replace(`/teams/${team}/`, '/teams/'));
		}
		if (player) {
			if (keepPlayerPage(param, value, player)) return switched(path);
			return switched(path.replace(`/players/${player}/`, '/players/'));
		}
		return switched(path);
	}

	// player_id 跨組別獨立；跨賽季則看目標賽季有沒有這位球員的登錄
	function keepPlayerPage(param: 'season' | 'group', value: string, player: string): boolean {
		if (param !== 'season') return false;
		const group = page.params.group;
		const cupId = findSeason(value)?.cup_id;
		if (!group || !isGroup(group) || cupId === undefined) return false;
		return findPlayerSeason(cupId, group, Number(player)) !== undefined;
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

<div class="wrap">
	<main>
		{@render children()}
	</main>
	<SiteFooter />
</div>

<style>
	.wrap {
		display: grid;
		gap: var(--space-7);
		padding: var(--space-4) var(--space-4) var(--space-7);
		min-inline-size: 0;
	}

	main {
		min-inline-size: 0;
	}

	@media (min-width: 64rem) {
		:global(body) {
			display: grid;
			grid-template-columns: var(--sidebar) minmax(0, 1fr);
			align-items: start;
		}

		.wrap {
			inline-size: 100%;
			max-inline-size: var(--content-max);
			padding: var(--space-6) var(--space-7);
		}
	}
</style>
