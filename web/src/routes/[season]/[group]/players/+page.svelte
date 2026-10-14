<script lang="ts">
	import EmptyState from '#lib/components/EmptyState.svelte';
	import PlayerLink from '#lib/components/PlayerLink.svelte';
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<svelte:head>
	<title>球員｜{data.season.name} {GROUP_LABEL[data.group]}｜TVL 數據</title>
</svelte:head>

<h1>球員</h1>

{#if data.teams.length === 0}
	<EmptyState title="這一季沒有週登錄資料" hint="有登錄資料後就會出現" />
{:else}
	{#each data.teams as { team, players } (team.team_id)}
		<section>
			<h2>
				<TeamBadge {team} size="sm" />
				{team.name}
			</h2>
			<ul class="players">
				{#each players as player (player.player_id)}
					<li>
						<PlayerLink season={data.season.name} group={data.group} player={{ ...player, team }} />
					</li>
				{/each}
			</ul>
		</section>
	{/each}
{/if}

<style>
	h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin-block: var(--space-6) var(--space-3);
	}

	h2 {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		font-size: var(--text-lg);
		font-weight: 700;
		margin-block: var(--space-6) var(--space-3);
	}

	.players {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: var(--space-3);
	}
</style>
