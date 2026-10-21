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
			class="chip"
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
		gap: var(--space-1);
	}

	@media (min-width: 64rem) {
		nav {
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(3.75rem, 1fr));
			column-gap: var(--space-1);
		}

		.chip {
			justify-content: center;
			min-block-size: 2.25rem;
			padding-inline: var(--space-2);
			white-space: nowrap;
			font-size: var(--text-sm);
			font-weight: 700;
		}
	}
</style>
