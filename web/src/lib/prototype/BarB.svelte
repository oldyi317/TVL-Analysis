<script lang="ts">
	import { page } from '$app/state';
	import SiteNav from '#lib/components/SiteNav.svelte';
	import type { Group } from '#lib/site.js';
	import BackButton from './BackButton.svelte';

	// 變體 B：吸頂 pill 列（深層頁最左多返回鈕）。其他變體與桌面版隱藏。
	let { season, group }: { season?: string; group?: Group } = $props();
	const deep = $derived(!!(page.params.game || page.params.team || page.params.player));
</script>

{#if season && group}
	<div class="bar">
		{#if deep}<BackButton />{/if}
		<div class="pills"><SiteNav {season} {group} /></div>
	</div>
{/if}

<style>
	.bar {
		display: none;
	}

	.pills {
		flex: 1;
		min-inline-size: 0;
	}

	@media (max-width: 63.99rem) {
		:global(html[data-shell='B']) .bar {
			display: flex;
			align-items: center;
			gap: var(--space-1);
			position: sticky;
			inset-block-start: var(--safe-top);
			z-index: 2;
			padding: 4px var(--space-4);
			background: var(--bg);
		}
	}
</style>
