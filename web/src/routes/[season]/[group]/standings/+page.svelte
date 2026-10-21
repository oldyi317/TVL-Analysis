<script lang="ts">
	import EmptyState from '#lib/components/EmptyState.svelte';
	import StandingsTable from '#lib/components/StandingsTable.svelte';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<svelte:head>
	<title>{data.season.name} {GROUP_LABEL[data.group]} 戰績｜TVL 數據</title>
</svelte:head>

<h1>{data.season.name} {GROUP_LABEL[data.group]} 戰績</h1>

{#if data.rows.length === 0}
	<EmptyState title="尚無戰績" hint="這一季開打後就會出現" />
{:else}
	<StandingsTable rows={data.rows} note={data.note ?? undefined} />
	<p class="basis">只計例行賽；黃金決勝局不計入</p>
{/if}

<style>
	h1 {
		margin-block: var(--space-2) var(--space-4);
	}

	.basis {
		margin: var(--space-3) 0 0;
		font-size: var(--text-sm);
		color: var(--text-muted);
	}
</style>
