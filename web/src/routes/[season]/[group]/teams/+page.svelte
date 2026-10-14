<script lang="ts">
	import { resolve } from '$app/paths';
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { Team } from '#lib/data/types.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	function teamHref(team: Team): string {
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

{#if data.teams.length === 0}
	<p class="empty">這一季尚無球隊資料</p>
{:else}
	<ul class="teams">
		{#each data.teams as team (team.team_id)}
			<li>
				{#if team.is_active}
					<a href={teamHref(team)}>
						<TeamBadge {team} />
						<span class="name">{team.name}</span>
					</a>
				{:else}
					<TeamBadge {team} />
					<span class="name">{team.name}</span>
				{/if}
			</li>
		{/each}
	</ul>
{/if}

<style>
	h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin-block: 1.5rem 0.75rem;
	}

	.teams {
		list-style: none;
		margin: 0;
		padding: 0;
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 0.75rem;
	}

	.teams li {
		display: flex;
		align-items: center;
		gap: 0.875rem;
		min-block-size: var(--tap);
		padding: 0.625rem 1rem;
	}

	.teams li:has(a) {
		padding: 0;
	}

	.teams a {
		display: flex;
		align-items: center;
		gap: 0.875rem;
		flex: 1;
		min-block-size: var(--tap);
		padding: 0.625rem 1rem;
		color: inherit;
		text-decoration: none;
	}

	.teams a:hover {
		background: var(--line);
	}

	.teams a:focus-visible {
		outline: 2px solid var(--focus);
		outline-offset: -2px;
	}

	.teams li + li {
		border-top: 1px solid var(--line);
	}

	.name {
		font-size: 1rem;
		font-weight: 500;
	}

	.empty {
		margin: 2rem 0;
		color: var(--text-muted);
	}
</style>
