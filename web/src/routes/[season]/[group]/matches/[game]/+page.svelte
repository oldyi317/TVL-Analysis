<script lang="ts">
	import { resolve } from '$app/paths';
	import BoxScoreTable from '#lib/components/BoxScoreTable.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import MatchCard from '#lib/components/MatchCard.svelte';
	import type { BoxRow } from '#lib/boxScore.js';
	import { stageLabel } from '#lib/match.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const match = $derived(data.match);
	const played = $derived(match.sets.length > 0);
	const homeTotal = $derived(match.sets.reduce((sum, [h]) => sum + h, 0));
	const awayTotal = $derived(match.sets.reduce((sum, [, a]) => sum + a, 0));
	const listHref = $derived(
		`${resolve('/[season]/[group]/matches', { season: data.season.name, group: data.group })}/#m-${match.date.slice(0, 7)}`
	);
</script>

{#snippet sideTables(sides: { home: BoxRow[]; away: BoxRow[] }, suffix: string)}
	<h3>{match.home.short_name}</h3>
	<BoxScoreTable rows={sides.home} label="{match.home.short_name} {suffix}" />
	<h3>{match.away.short_name}</h3>
	<BoxScoreTable rows={sides.away} label="{match.away.short_name} {suffix}" />
{/snippet}

<svelte:head>
	<title>{match.home.short_name} 對 {match.away.short_name} {match.date}｜{data.season.name} {GROUP_LABEL[data.group]} 賽事｜TVL 數據</title>
</svelte:head>

<a class="back" href={listHref}>
	<svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
		<path d="M10 3 5 8l5 5" />
	</svg>
	賽事列表
</a>

<h1>{match.home.short_name} 對 {match.away.short_name}</h1>

<MatchCard {match} stage={stageLabel(match.stage, match.week)} showSets={false} />

{#if played}
	<h2>逐局比分</h2>
	<table>
		<thead>
			<tr>
				<th scope="col">局</th>
				<th scope="col">{match.home.short_name}</th>
				<th scope="col">{match.away.short_name}</th>
			</tr>
		</thead>
		<tbody>
			{#each match.sets as [h, a], i (i)}
				<tr>
					<th scope="row">第 {i + 1} 局</th>
					<td class:strong={h > a}>{h}</td>
					<td class:strong={a > h}>{a}</td>
				</tr>
			{/each}
			<tr class="total">
				<th scope="row">合計</th>
				<td>{homeTotal}</td>
				<td>{awayTotal}</td>
			</tr>
			{#if match.golden_set}
				<tr class="golden">
					<th scope="row">黃金決勝局</th>
					<td class:strong={match.golden_set[0] > match.golden_set[1]}>{match.golden_set[0]}</td>
					<td class:strong={match.golden_set[1] > match.golden_set[0]}>{match.golden_set[1]}</td>
				</tr>
			{/if}
		</tbody>
	</table>
	{#if match.golden_set}
		<p class="basis">黃金決勝局是總決賽加打的單局，不計入局數與戰績</p>
	{/if}

	<h2>單場數據</h2>
	{#if data.boxScore}
		{@render sideTables(data.boxScore, '單場數據')}
		<p class="basis">接發、防守、舉球欄為「到位／次數」</p>
		{#if data.boxScore.golden_set}
			<h2>黃金決勝局數據</h2>
			{@render sideTables(data.boxScore.golden_set, '黃金決勝局數據')}
		{/if}
	{:else}
		<EmptyState title="這場沒有單場數據" hint="資料來源沒有提供這場的技術統計" />
	{/if}
{:else}
	<EmptyState title="這場還沒開打" hint="開打後會有逐局比分" />
{/if}

<style>
	.back {
		display: inline-flex;
		align-items: center;
		gap: var(--space-1);
		min-block-size: var(--tap);
		margin-block-start: var(--space-2);
		font-weight: 500;
	}

	.back svg {
		flex: none;
	}

	h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin-block: 0 var(--space-3);
	}

	h2 {
		font-size: var(--text-lg);
		font-weight: 700;
		margin-block: var(--space-6) var(--space-3);
	}

	h3 {
		font-size: var(--text-md);
		font-weight: 700;
		margin-block: var(--space-4) var(--space-2);
	}

	table {
		inline-size: 100%;
		border-collapse: collapse;
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: var(--radius-md);
		overflow: hidden;
	}

	th,
	td {
		padding: var(--space-2) var(--space-4);
		border-block-start: 1px solid var(--line);
		text-align: end;
	}

	thead th {
		border-block-start: 0;
		font-size: var(--text-sm);
		font-weight: 500;
		color: var(--text-muted);
	}

	th[scope='row'] {
		text-align: start;
		font-size: var(--text-md);
		font-weight: 400;
		color: var(--text-muted);
	}

	td {
		font-size: var(--num-md);
		font-weight: 400;
		line-height: 1.2;
		color: var(--text-muted);
	}

	td.strong {
		font-weight: var(--weight-num);
		color: var(--text);
	}

	.total td {
		font-size: var(--text-md);
		color: var(--text);
	}

	.golden th,
	.golden td {
		border-block-start: 2px solid var(--line);
	}

	.golden td {
		font-size: var(--text-md);
	}

	.basis {
		margin: var(--space-3) 0 0;
		font-size: var(--text-sm);
		color: var(--text-muted);
	}
</style>
