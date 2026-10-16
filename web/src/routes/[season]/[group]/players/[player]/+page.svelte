<script lang="ts">
	import { resolve } from '$app/paths';
	import { headlineStats } from '#lib/advanced.js';
	import AdvancedStats from '#lib/components/AdvancedStats.svelte';
	import BigStat from '#lib/components/BigStat.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import { findTeam, GROUP_LABEL, playerDisplayName, positionLabel } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const name = $derived(playerDisplayName(data.player));
	const stats = $derived(data.stats);
	const headline = $derived(headlineStats(stats));
	const listHref = $derived(
		`${resolve('/[season]/[group]/players', { season: data.season.name, group: data.group })}/`
	);
	const teamHref = $derived(
		findTeam(data.season.cup_id, data.group, data.team.team_id)
			? `${resolve('/[season]/[group]/teams/[team]', {
					season: data.season.name,
					group: data.group,
					team: String(data.team.team_id)
				})}/`
			: undefined
	);
	const profile = $derived([
		['背號', stats.jersey_number === null ? '—' : String(stats.jersey_number)],
		['位置', positionLabel(stats.position, true)],
		['身高', data.player.height_cm === null ? '—' : `${data.player.height_cm} cm`],
		['生日', data.player.dob ?? '—']
	]);
</script>

<svelte:head>
	<title>{name}｜{data.season.name} {GROUP_LABEL[data.group]} 球員｜TVL 數據</title>
</svelte:head>

<a class="back" href={listHref}>
	<svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
		<path d="M10 3 5 8l5 5" />
	</svg>
	球員列表
</a>

<header>
	<TeamBadge team={data.team} size="lg" />
	<div>
		<h1>{name}</h1>
		{#if teamHref}
			<a class="team" href={teamHref}>{data.team.name}</a>
		{:else}
			<p class="team">{data.team.name}</p>
		{/if}
	</div>
</header>

<h2>基本資料</h2>
<dl class="profile">
	{#each profile as [label, value] (label)}
		<div>
			<dt>{label}</dt>
			<dd>{value}</dd>
		</div>
	{/each}
</dl>

<div class="stats-head">
	<h2>本季數據</h2>
	{#if stats.games > 0 && headline.fewSets}
		<span class="few-sets">樣本少</span>
	{/if}
</div>
{#if stats.games === 0}
	<EmptyState title="這一季還沒有單場數據" hint="有單場數據後就會出現" />
{:else}
	<div class="stats">
		{#each headline.items as item (item.label)}
			<BigStat label={item.label} value={item.value} detail={item.detail} />
		{/each}
	</div>
	<dl class="totals">
		<div>
			<dt>出賽場次</dt>
			<dd>{stats.games}</dd>
		</div>
		<div>
			<dt>出賽局數</dt>
			<dd>{stats.sets_played}</dd>
		</div>
		<div>
			<dt>總得分</dt>
			<dd>{stats.total_points}</dd>
		</div>
	</dl>
{/if}

<AdvancedStats advanced={data.advanced} distribution={data.distribution} gameLog={data.gameLog} />

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

	header {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		margin-block-end: var(--space-3);
	}

	h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin: 0;
	}

	.team {
		display: inline-flex;
		align-items: center;
		min-block-size: var(--tap);
		margin: 0;
		color: var(--text-muted);
	}

	h2 {
		font-size: var(--text-lg);
		font-weight: 700;
		margin-block: var(--space-6) var(--space-3);
	}

	.profile,
	.stats,
	.totals {
		display: grid;
		gap: var(--space-2);
		margin: 0;
		padding: var(--space-3);
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: var(--radius-md);
	}

	.profile {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.stats,
	.totals {
		grid-template-columns: repeat(3, minmax(0, 1fr));
	}

	.totals {
		margin-block-start: var(--space-2);
	}

	.stats-head {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
	}

	.stats > :global(div) {
		container-type: inline-size;
	}

	.stats :global(p.value) {
		font-size: min(var(--num-xl), 24cqi);
	}

	.few-sets {
		padding-inline: var(--space-2);
		font-size: var(--text-sm);
		color: var(--text-muted);
		border: 1px solid var(--line);
		border-radius: var(--radius-md);
	}

	dt {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	dd {
		margin: 0;
		font-size: var(--text-lg);
		font-weight: 700;
		overflow-wrap: anywhere;
	}
</style>
