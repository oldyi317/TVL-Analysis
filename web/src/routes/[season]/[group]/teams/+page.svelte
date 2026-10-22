<script lang="ts">
	import { resolve } from '$app/paths';
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import { GROUP_LABEL, type SeasonTeam } from '#lib/site.js';
	import type { StandingsEntry } from '#lib/data/types.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	function teamHref(team: SeasonTeam): string {
		return `${resolve('/[season]/[group]/teams/[team]', {
			season: data.season.name,
			group: data.group,
			team: String(team.team_id)
		})}/`;
	}
</script>

<svelte:head>
	<title>{data.season.name} {GROUP_LABEL[data.group]} 球隊｜TVL 數據</title>
</svelte:head>

<h1>{data.season.name} {GROUP_LABEL[data.group]} 球隊</h1>

{#snippet body(team: SeasonTeam, standing: StandingsEntry | null)}
	<TeamBadge {team} size="lg" />
	<span class="text">
		<span class="name">{team.name}</span>
		{#if team.short_name !== team.name}
			<span class="short">{team.short_name}</span>
		{/if}
		{#if standing}
			<span class="record">第 {standing.rank} 名・{standing.wins} 勝 {standing.losses} 敗</span>
		{/if}
	</span>
{/snippet}

{#if data.teams.length === 0}
	<p class="empty">這一季尚無球隊資料</p>
{:else}
	<ul class="teams">
		{#each data.teams as { team, standing }, i (team.team_id)}
			<li class="rise" style:--i={Math.min(i, 5)}>
				<a class="card" href={teamHref(team)}>{@render body(team, standing)}</a>
			</li>
		{/each}
	</ul>
{/if}

<style>
	h1 {
		margin-block: var(--space-2) var(--space-4);
	}

	.teams {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--space-3);
	}

	@media (min-width: 40rem) {
		.teams {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}

	@media (min-width: 64rem) {
		.teams {
			grid-template-columns: repeat(3, minmax(0, 1fr));
		}
	}

	.card {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		min-block-size: var(--tap);
	}

	.text {
		display: grid;
		min-inline-size: 0;
	}

	.name {
		font-weight: 700;
		overflow-wrap: anywhere;
	}

	.record {
		margin-block-start: var(--space-1);
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.short {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.empty {
		margin: var(--space-6) 0;
		color: var(--text-muted);
	}
</style>
