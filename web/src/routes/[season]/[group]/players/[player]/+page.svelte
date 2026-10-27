<script lang="ts">
	import { resolve } from '$app/paths';
	import { headlineStats } from '#lib/advanced.js';
	import AdvancedStats from '#lib/components/AdvancedStats.svelte';
	import BigStat from '#lib/components/BigStat.svelte';
	import Icon from '#lib/components/Icon.svelte';
	import DataNote from '#lib/components/DataNote.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import { displayPosition, findTeam, GROUP_LABEL, playerDisplayName, positionLabel } from '#lib/site.js';
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
		['位置', positionLabel(displayPosition(stats), true)],
		['身高', data.player.height_cm === null ? '—' : `${data.player.height_cm} cm`],
		['體重', data.player.weight_kg === null ? '—' : `${data.player.weight_kg} kg`]
	]);
</script>

<svelte:head>
	<title>{name}｜{data.season.name} {GROUP_LABEL[data.group]} 球員｜TVL 數據</title>
</svelte:head>

<a class="back" href={listHref}>
	<Icon name="chevron-left" size={16} />
	球員列表
</a>

<header>
	<h1>{name}</h1>
	<p class="team-line">
		<TeamBadge team={data.team} size="sm" />
		{#if teamHref}
			<a class="team" href={teamHref}>{data.team.name}</a>
		{:else}
			<span class="team">{data.team.name}</span>
		{/if}
	</p>
	<p class="meta">
		背號 {stats.jersey_number === null ? '—' : stats.jersey_number}・{positionLabel(displayPosition(stats), true)}
	</p>
</header>

<h2>球員資料</h2>
<dl class="profile card">
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
	<EmptyState title="本季尚無上場紀錄" hint="有單場數據後就會出現" />
{:else}
	{#if stats.sets_played === null}
		<DataNote text="官網單場數據不提供出賽局數" />
	{/if}
	<div class="stats">
		{#each headline.items as item, i (item.label)}
			<BigStat label={item.label} value={item.value} detail={item.detail} tone={(i % 3) + 1 as 1 | 2 | 3} />
		{/each}
	</div>
	<dl class="totals card">
		<div>
			<dt>出賽場次</dt>
			<dd>{stats.games}</dd>
		</div>
		<div>
			<dt>出賽局數</dt>
			<dd>{stats.sets_played ?? '—'}</dd>
		</div>
		<div>
			<dt>總得分</dt>
			<dd>{stats.total_points}</dd>
		</div>
	</dl>
{/if}

<AdvancedStats setsPlayed={stats.sets_played} advanced={data.advanced} distribution={data.distribution} gameLog={data.gameLog} />

<style>
	.back {
		display: inline-flex;
		align-items: center;
		gap: var(--space-1);
		min-block-size: var(--tap);
		color: var(--accent);
		font-weight: 600;
	}

	.back:hover {
		color: var(--accent-strong);
		text-decoration: underline;
	}

	header {
		margin-block: var(--space-2) var(--space-2);
	}

	.team-line {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin: var(--space-2) 0 0;
	}

	.team {
		font-weight: 600;
	}

	a.team {
		color: var(--accent);
	}

	a.team:hover {
		color: var(--accent-strong);
		text-decoration: underline;
	}

	.meta {
		margin: var(--space-1) 0 0;
		color: var(--text-muted);
		font-size: var(--text-sm);
	}

	h2 {
		margin-block: var(--space-5) var(--space-3);
	}

	.profile,
	.totals {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: var(--space-3) var(--space-4);
		margin: 0;
	}

	.totals {
		grid-template-columns: repeat(3, minmax(0, 1fr));
		margin-block-start: var(--space-3);
	}

	.stats {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: var(--space-2);
	}

	.stats :global(.stat-tile) {
		padding: var(--space-3);
	}

	.stats :global(p.value) {
		font-size: var(--num-md);
	}

	@media (min-width: 40rem) {
		.stats {
			gap: var(--space-3);
		}

		.stats :global(.stat-tile) {
			padding: var(--space-4);
		}

		.stats :global(p.value) {
			font-size: var(--num-lg);
		}
	}

	.stats-head {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
	}

	.few-sets {
		padding: 2px 10px;
		border-radius: var(--radius-full);
		background: var(--surface-2);
		font-size: var(--text-sm);
		font-weight: 700;
		color: var(--text-muted);
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
