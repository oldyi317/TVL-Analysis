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
			<ul class="players card">
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
		margin-block: var(--space-4) var(--space-2);
	}

	section {
		margin-block-start: var(--space-5);
	}

	h2 {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin-block-end: var(--space-3);
	}

	.players {
		list-style: none;
		margin: 0;
		padding: var(--space-2);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 2px var(--space-3);
	}

	@media (min-width: 40rem) {
		.players {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}

	@media (min-width: 64rem) {
		.players {
			grid-template-columns: repeat(3, minmax(0, 1fr));
		}
	}
</style>
