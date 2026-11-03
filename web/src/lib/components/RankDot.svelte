<script lang="ts">
	let { rank }: { rank: number } = $props();
</script>

<span class="rank-dot" class:top={rank <= 3}>{rank}</span>

<style>
	.rank-dot {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		inline-size: 1.5rem;
		block-size: 1.5rem;
		border-radius: 50%;
		color: var(--text-muted);
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}

	.rank-dot.top {
		background: var(--accent-tint);
		color: var(--accent);
		font-weight: 700;
	}
</style>
