<script lang="ts">
	import { resolve } from '$app/paths';
	import type { Group } from '#lib/site.js';
	import GroupSwitch from './GroupSwitch.svelte';
	import SeasonSwitch from './SeasonSwitch.svelte';
	import SiteNav from './SiteNav.svelte';

	let {
		season,
		group,
		hrefFor
	}: {
		season?: string;
		group?: Group;
		hrefFor: (param: 'season' | 'group', value: string) => string | undefined;
	} = $props();
</script>

<header>
	<div class="top">
		<a class="brand" href={resolve('/')}>TVL 數據</a>
		<span class="note">非官方・資料整理自聯賽官網</span>
	</div>
	{#if season && group}
		<div class="switches">
			<GroupSwitch current={group} hrefFor={(g) => hrefFor('group', g)} />
			<SeasonSwitch current={season} hrefFor={(s) => hrefFor('season', s)} />
			<SiteNav {season} {group} />
		</div>
	{/if}
</header>

<style>
	header {
		display: grid;
		gap: var(--space-3);
		padding: var(--space-3) var(--space-4) var(--space-4);
		background: var(--surface);
		border-block-end: 1px solid var(--card-edge);
		box-shadow: var(--shadow-card);
	}

	.top {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		column-gap: var(--space-3);
	}

	.brand {
		font-size: var(--text-lg);
		font-weight: 800;
		color: var(--text);
	}

	.note {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.switches {
		display: grid;
		gap: var(--space-3);
		min-inline-size: 0;
	}

	@media (min-width: 64rem) {
		header {
			position: sticky;
			inset-block-start: 0;
			block-size: 100dvh;
			align-content: start;
			gap: var(--space-5);
			padding: var(--space-5);
			overflow-y: auto;
			border-block-end: 0;
			border-inline-end: 1px solid var(--card-edge);
			box-shadow: none;
		}

		.top {
			display: grid;
			gap: var(--space-1);
		}

		.brand {
			font-size: var(--text-xl);
		}

		.switches {
			gap: var(--space-5);
		}
	}
</style>
