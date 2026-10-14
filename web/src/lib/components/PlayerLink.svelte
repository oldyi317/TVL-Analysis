<script lang="ts">
	import { resolve } from '$app/paths';
	import type { Team } from '#lib/data/types.js';
	import type { Group } from '#lib/site.js';
	import PlayerCard from './PlayerCard.svelte';

	let {
		season,
		group,
		player
	}: {
		season: string;
		group: Group;
		player: {
			player_id: number;
			name: string;
			number: number | null;
			position: string | null;
			team: Team;
		};
	} = $props();

	const href = $derived(
		`${resolve('/[season]/[group]/players/[player]', {
			season,
			group,
			player: String(player.player_id)
		})}/`
	);
</script>

<a class="player-link" {href}>
	<PlayerCard {player} />
</a>

<style>
	.player-link {
		display: block;
		border-radius: var(--radius-md);
	}

	.player-link:focus-visible {
		outline: 2px solid var(--focus);
		outline-offset: 2px;
	}

	/* 整張卡是連結：指標停留與按下時外框加深 */
	@media (hover: hover) {
		.player-link:hover :global(article) {
			border-color: var(--text-muted);
		}
	}

	.player-link:active :global(article) {
		border-color: var(--text);
	}
</style>
