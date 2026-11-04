<script lang="ts">
	import type { Snippet } from 'svelte';
	import Chart from '#lib/chart/Chart.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';

	let {
		title,
		games,
		conclusion,
		option,
		emptyTitle,
		note,
		height,
		label,
		heading = 'h3',
		controls,
		table
	}: {
		title: string;
		games: number | readonly { name: string; games: number }[];
		conclusion: string | null;
		option: object | null;
		emptyTitle: string;
		note?: string;
		height?: string;
		label?: string;
		heading?: 'h2' | 'h3';
		controls?: Snippet;
		table: Snippet;
	} = $props();

	const uid = $props.id();
	const tableId = `${uid}-table`;
	let showTable = $state(false);
	const gamesText = $derived.by(() => {
		if (typeof games === 'number') return `已賽 ${games} 場`;
		if (games.every((t) => t.games === games[0].games)) return `已賽 ${games[0].games} 場`;
		return games.map((t) => `${t.name} 已賽 ${t.games} 場`).join('・');
	});
	const ariaLabel = $derived(label ?? (conclusion ? `${title}：${conclusion}` : `${title}，${gamesText}`));
</script>

<section class="card" aria-label={title}>
	<div class="head">
		<svelte:element this={heading} class="title">{title}</svelte:element>
		{#if option}
			<span class="games">{gamesText}</span>
		{/if}
	</div>
	{@render controls?.()}
	{#if !option}
		<EmptyState title={emptyTitle} />
	{:else}
		{#if conclusion}
			<p class="conclusion">{conclusion}</p>
		{/if}
		<Chart {option} {ariaLabel} {height} />
		{#if note}
			<p class="note">{note}</p>
		{/if}
		<button
			type="button"
			class="toggle"
			aria-expanded={showTable ? 'true' : 'false'}
			aria-controls={tableId}
			onclick={() => (showTable = !showTable)}
		>
			以表格檢視
		</button>
		<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
		<div id={tableId} class="table" hidden={!showTable} tabindex={showTable ? 0 : -1}>
			{#if showTable}
				{@render table()}
			{/if}
		</div>
	{/if}
</section>

<style>
	section {
		margin-block-end: var(--space-4);
	}

	.head {
		display: flex;
		align-items: baseline;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: var(--space-2);
		margin-block-end: var(--space-2);
	}

	.title {
		margin: 0;
		font-size: var(--text-md);
		font-weight: 700;
	}

	.games {
		flex: none;
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.conclusion {
		margin: 0 0 var(--space-2);
		font-weight: 600;
	}

	.note {
		margin: var(--space-2) 0 0;
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.toggle {
		min-block-size: var(--tap);
		margin-block-start: var(--space-2);
		padding: 0;
		background: none;
		border: 0;
		color: var(--accent);
		font: inherit;
		font-weight: 600;
		cursor: pointer;
	}

	.table {
		overflow-x: auto;
		margin-block-start: var(--space-2);
	}

	.table :global(table) {
		inline-size: 100%;
		border-collapse: collapse;
		font-size: var(--text-sm);
		white-space: nowrap;
	}

	.table :global(th),
	.table :global(td) {
		padding: var(--space-2) var(--space-2);
		border-block-start: 1px solid var(--line);
		text-align: start;
		vertical-align: top;
	}

	.table :global(thead th) {
		border-block-start: 0;
		color: var(--text-muted);
		font-weight: 600;
	}

	.table :global(tbody th) {
		font-weight: 600;
	}

	.table :global(.num) {
		text-align: end;
		font-variant-numeric: tabular-nums;
	}

	.table :global(.rank),
	.table :global(.self) {
		color: var(--text-muted);
		font-size: var(--text-sm);
		font-weight: 400;
	}

	.table :global(.rank) {
		display: block;
	}

	.toggle:hover {
		color: var(--accent-strong);
		text-decoration: underline;
	}
</style>
