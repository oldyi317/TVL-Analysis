<script lang="ts">
	import { resolve } from '$app/paths';
	import type { Group, SeasonTeam } from '#lib/site.js';
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
			team: SeasonTeam;
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
		transition: background-color 160ms ease-out;
	}

	@media (hover: hover) {
		.player-link:hover {
			background: var(--accent-tint);
		}
	}

	.player-link:active {
		background: var(--accent-tint);
	}
</style>
