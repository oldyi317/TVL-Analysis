<script lang="ts">
	import { resolve } from '$app/paths';
	import { DEFAULT_GROUP } from '#lib/groupMemory.js';
	import { currentSeason, GROUPS, GROUP_LABEL, type Group } from '#lib/site.js';
	import SegmentNav from './SegmentNav.svelte';
	import SeasonSwitch from './SeasonSwitch.svelte';
	import SiteNav from './SiteNav.svelte';

	let {
		season,
		group,
		records = false,
		navOnly = false,
		hrefFor
	}: {
		season?: string;
		group?: Group;
		records?: boolean;
		navOnly?: boolean;
		hrefFor: (param: 'season' | 'group', value: string) => string | undefined;
	} = $props();
</script>

<header>
	<div class="top">
		<a class="brand" href={resolve('/')}>VStat 企排</a>
		<span class="note">非官方・資料整理自聯賽官網</span>
	</div>
	{#if season && group}
		<div class="switches">
			<SegmentNav
				label="組別"
				items={GROUPS.map((g) => ({ key: g, label: GROUP_LABEL[g], href: hrefFor('group', g), current: g === group }))}
			/>
			{#if !records}
				<SeasonSwitch current={season} hrefFor={(s) => hrefFor('season', s)} />
			{/if}
			<SiteNav {season} {group} />
		</div>
	{:else if navOnly}
		<!-- 不含賽季與組別的頁只留導覽，連結基準比照手機分頁列：現行賽季與預設組別 -->
		<SiteNav season={currentSeason.name} group={DEFAULT_GROUP} />
	{/if}
</header>

<style>
	/* 桌面側欄；手機改由 TopBar 與 TabBar 負責 */
	header {
		display: none;
	}

	.note {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.brand {
		font-size: var(--text-xl);
		font-weight: 800;
		color: var(--text);
	}

	.top {
		display: grid;
		gap: var(--space-1);
	}

	.switches {
		display: grid;
		gap: var(--space-5);
		min-inline-size: 0;
	}

	@media (min-width: 64rem) {
		header {
			position: sticky;
			inset-block-start: 0;
			display: grid;
			block-size: 100dvh;
			align-content: start;
			gap: var(--space-5);
			padding: var(--space-5);
			overflow-y: auto;
			background: var(--surface);
			border-inline-end: 1px solid var(--card-edge);
		}
	}
</style>
