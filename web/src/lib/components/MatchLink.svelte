<script lang="ts">
	import type { StyleMatch } from '#lib/match.js';
	import MatchCard from './MatchCard.svelte';

	let {
		href,
		match,
		stage,
		size = 'md'
	}: { href: string; match: StyleMatch; stage?: string | null; size?: 'md' | 'lg' } = $props();
</script>

<a class="card-link" {href}>
	<MatchCard {match} {stage} {size} />
</a>

<style>
	.card-link {
		display: block;
		border-radius: var(--radius-lg);
	}

	.card-link :global(article) {
		transition:
			transform 160ms ease-out,
			box-shadow 160ms ease-out;
	}

	@media (hover: hover) {
		.card-link:hover :global(article) {
			transform: translateY(-2px);
			box-shadow: var(--shadow-pop);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.card-link:hover :global(article) {
			transform: none;
		}
	}
</style>
