<script lang="ts">
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import { resolve } from '$app/paths';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<svelte:head>
	<title>{data.season.name} {GROUP_LABEL[data.group]} 球隊｜TVL 數據</title>
</svelte:head>

<h1>{data.season.name} {GROUP_LABEL[data.group]} 球隊</h1>

<a class="to-standings" href="{resolve('/[season]/[group]/standings', { season: data.season.name, group: data.group })}/">戰績</a>

{#if data.teams.length === 0}
	<p class="empty">這一季尚無球隊資料</p>
{:else}
	<ul class="teams">
		{#each data.teams as team (team.team_id)}
			<li>
				<TeamBadge {team} />
				<span class="name">{team.name}</span>
			</li>
		{/each}
	</ul>
{/if}

<style>
	h1 {
		font-size: 1.375rem;
		font-weight: 700;
		margin-block: 1.5rem 0.75rem;
	}

	.to-standings {
		display: inline-flex;
		align-items: center;
		min-block-size: var(--tap);
		margin-block-end: var(--space-2);
		font-weight: 500;
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
