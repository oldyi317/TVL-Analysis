<script lang="ts">
	import { resolve } from '$app/paths';
	import type { Group } from '#lib/site.js';
	import GroupSwitch from './GroupSwitch.svelte';
	import SeasonSwitch from './SeasonSwitch.svelte';

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
	<div class="wrap top">
		<a class="brand" href={resolve('/')}>TVL 數據</a>
		<span class="note">非官方・資料整理自聯賽官網</span>
	</div>
	{#if season && group}
		<div class="wrap switches">
			<GroupSwitch current={group} hrefFor={(g) => hrefFor('group', g)} />
			<SeasonSwitch current={season} hrefFor={(s) => hrefFor('season', s)} />
		</div>
	{/if}
</header>

<style>
	header {
		background: var(--surface);
		border-bottom: 1px solid var(--line);
	}

	.wrap {
		max-inline-size: var(--measure);
		margin-inline: auto;
		padding-inline: var(--space-4);
	}

	.top {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		column-gap: var(--space-3);
	}

	.brand {
		display: inline-flex;
		align-items: center;
		min-block-size: var(--tap);
		font-size: var(--text-lg);
		font-weight: 700;
	}

	.note {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.switches {
		display: grid;
		gap: var(--space-2);
		padding-block-end: var(--space-2);
	}
</style>
