<script lang="ts">
	import { seasons } from '#lib/site.js';

	let {
		current,
		hrefFor
	}: { current: string; hrefFor: (season: string) => string | undefined } = $props();
</script>

<nav aria-label="賽季">
	{#each seasons as season (season.cup_id)}
		<a
			href={hrefFor(season.name)}
			aria-current={current === season.name ? 'page' : undefined}
		>
			{season.name}
		</a>
	{/each}
</nav>

<style>
	nav {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-2);
	}

	a {
		display: inline-flex;
		align-items: center;
		min-block-size: var(--tap);
		padding-inline: var(--space-3);
		font-size: var(--text-md);
		color: var(--text-muted);
		border: 1px solid transparent;
		border-radius: var(--radius-sm);
	}

	a[aria-current='page'] {
		color: var(--text);
		background: var(--surface);
		border-color: var(--text);
	}
</style>
