<script lang="ts">
	import { positionLabel, type SeasonTeam } from '#lib/site.js';

	let {
		player
	}: { player: { name: string; number: number | null; position: string | null; team: SeasonTeam } } =
		$props();
</script>

<article>
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
		gap: var(--space-3);
		min-block-size: 3.5rem;
		padding: var(--space-2) var(--space-3);
	}

	.number {
		display: flex;
		align-items: center;
		justify-content: center;
		flex: none;
		inline-size: 2.5rem;
		block-size: 2.5rem;
		border-radius: var(--radius-full);
		background: var(--surface-2);
		color: var(--text);
		font-size: var(--text-md);
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
