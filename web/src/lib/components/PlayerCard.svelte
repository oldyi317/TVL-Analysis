<script lang="ts">
	import type { Team } from '#lib/data/types.js';
	import { positionLabel } from '#lib/site.js';
	import { teamStyle } from './teamStyle.js';

	let {
		player
	}: { player: { name: string; number: number | null; position: string | null; team: Team } } =
		$props();
</script>

<article style={teamStyle(player.team)}>
	<span class="number">{player.number ?? '—'}</span>
	<div class="info">
		<p class="name">{player.name}</p>
		{#if player.position}
			<p class="position">{positionLabel(player.position, true)}</p>
		{/if}
	</div>
</article>

<style>
	article {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		padding: var(--space-4);
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: var(--radius-md);
	}

	.number {
		display: flex;
		align-items: center;
		justify-content: center;
		flex: none;
		inline-size: 3.5rem;
		block-size: 3.5rem;
		border-radius: var(--radius-full);
		background: var(--team-fill);
		color: var(--team-on-fill);
		font-size: var(--num-md);
		font-weight: var(--weight-num);
		line-height: 1;
	}

	p {
		margin: 0;
	}

	.name {
		font-size: var(--text-md);
		font-weight: 700;
	}

	.position {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}
</style>
