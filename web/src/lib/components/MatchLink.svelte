<script lang="ts">
	import type { StyleMatch } from '#lib/match.js';
	import MatchCard from './MatchCard.svelte';

	let { href, match, stage }: { href: string; match: StyleMatch; stage?: string | null } = $props();
</script>

<a class="card-link" {href}>
	<MatchCard {match} {stage} />
</a>

<style>
	.card-link {
		display: block;
		border-radius: var(--radius-md);
	}

	.card-link:focus-visible {
		outline: 2px solid var(--focus);
		outline-offset: 2px;
	}

	/* 整張卡是連結：指標停留與按下時外框加深；只動上下與右側，左側隊色條不變 */
	@media (hover: hover) {
		.card-link:hover :global(article) {
			border-block-color: var(--text-muted);
			border-inline-end-color: var(--text-muted);
		}
	}

	.card-link:active :global(article) {
		border-block-color: var(--text);
		border-inline-end-color: var(--text);
	}
</style>
