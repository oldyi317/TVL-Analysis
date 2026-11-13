<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { afterNavigate } from '$app/navigation';
	import { asset, resolve } from '$app/paths';
	import SiteFooter from '#lib/components/SiteFooter.svelte';
	import SiteHeader from '#lib/components/SiteHeader.svelte';
	import TabBar from '#lib/components/TabBar.svelte';
	import TopBar from '#lib/components/TopBar.svelte';
	import { browserStorage, writeGroup } from '#lib/groupMemory.js';
	import { detailListOf, listHref, nextDepth } from '#lib/nav.js';
	import { parseId, resolveSeasonGroup } from '#lib/route.js';
	import { currentSeason, findPlayerSeason, findTeam, isGroup, type Group } from '#lib/site.js';
	import type { LayoutProps } from './$types';
	import type { PageData as MatchPageData } from './[season]/[group]/matches/[game]/$types';

	let { children }: LayoutProps = $props();

	// 切換賽季或組別：只替換路徑中該片段的第一次出現，停留在同一頁面類型。
	// 站台在網域根目錄（base 是空字串），賽季與組別是路徑的前兩段，不會與其他片段撞名。
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

	// 切換賽季時，目標賽季在同組別下找不找得到這個 id；切換組別一律不保留
	function keepDetailPage(
		param: 'season' | 'group',
		value: string,
		raw: string,
		find: (cupId: number, group: Group, id: number) => unknown
	): boolean {
		if (param !== 'season') return false;
		const target = resolveSeasonGroup({ season: value, group: page.params.group });
		const id = parseId(raw);
		if (!target || id === undefined) return false;
		return find(target.season.cup_id, target.group, id) !== undefined;
	}

	// player_id 跨組別獨立；跨賽季則看目標賽季有沒有這位球員的登錄
	function keepPlayerPage(param: 'season' | 'group', value: string, player: string): boolean {
		return keepDetailPage(param, value, player, findPlayerSeason);
	}

	// team_id 跨組別是不同隊；跨賽季則看目標賽季有沒有這支現役隊
	function keepTeamPage(param: 'season' | 'group', value: string, team: string): boolean {
		return keepDetailPage(param, value, team, findTeam);
	}

	const group = $derived(page.params.group && isGroup(page.params.group) ? page.params.group : undefined);

	// 歷史紀錄頁不屬於任何賽季：導覽與組別切換沿用現行賽季當連結基準，不顯示賽季切換
	const onRecords = $derived(page.route.id?.startsWith('/records/') ?? false);
	const season = $derived(page.params.season ?? (onRecords ? currentSeason.name : undefined));

	// 站內瀏覽深度只留在記憶體：進站歸零，重新整理後也歸零
	let depth = $state(0);
	afterNavigate((nav) => {
		depth = nextDepth(depth, nav.type, 'delta' in nav ? nav.delta : 0);
	});

	const back = $derived.by(() => {
		const section = detailListOf(page.route.id);
		const detailSeason = page.params.season;
		if (!section || !detailSeason || !group) return undefined;
		const date = section === 'matches' ? (page.data as MatchPageData).match.date : undefined;
		const href = listHref(section, `${resolve('/[season]/[group]', { season: detailSeason, group })}/`, date);
		return { href, canGoBack: depth > 0 };
	});

	$effect(() => {
		if (group) writeGroup(browserStorage(), group);
	});

	// e2e 等到這個記號才互動：根 layout 掛載時整頁已 hydrate，SvelteKit 也在同一輪事件迴圈內接手站內連結
	onMount(() => {
		document.documentElement.dataset.hydrated = '';
	});
</script>

<svelte:head>
	<link rel="icon" type="image/png" href={asset('favicon.png')} />
	<link rel="manifest" href={resolve('/manifest.webmanifest')} />
	<link rel="apple-touch-icon" href={asset('icons/apple-touch-icon.png')} />
</svelte:head>

<TopBar {season} {group} records={onRecords} {back} hrefFor={switchHref} />
<SiteHeader {season} {group} records={onRecords} hrefFor={switchHref} />

<div class="wrap">
	<main>
		{@render children()}
	</main>
	<SiteFooter />
</div>

<TabBar {season} {group} />

<style>
	.wrap {
		display: grid;
		gap: var(--space-7);
		padding: var(--space-4) calc(var(--space-4) + var(--safe-right))
			calc(var(--space-7) + var(--tabbar-h) + var(--safe-bottom)) calc(var(--space-4) + var(--safe-left));
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
