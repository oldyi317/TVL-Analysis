<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { hrefOf, NAV_ITEMS, sectionOf } from '#lib/nav.js';
	import type { Group } from '#lib/site.js';
	import Icon from './Icon.svelte';

	let { season, group }: { season: string; group: Group } = $props();

	const base = $derived(`${resolve('/[season]/[group]', { season: encodeURIComponent(season), group })}/`);
	const recordsBase = $derived(`${resolve('/records/[group]/[[kind=recordKind]]', { group })}/`);
	const section = $derived(sectionOf(page.route.id));
</script>

<nav aria-label="頁面">
	{#each NAV_ITEMS as item (item.label)}
		<a href={hrefOf(item.section, base, recordsBase)} aria-current={section === item.section ? 'page' : undefined}>
			<Icon name={item.icon} />
			<span>{item.label}</span>
		</a>
	{/each}
</nav>

<style>
	nav {
		display: none;
	}

	a {
		position: relative;
		display: flex;
		align-items: center;
		gap: var(--space-3);
		min-block-size: var(--tap);
		padding-inline: var(--space-3);
		font-size: var(--text-md);
		font-weight: 700;
		color: var(--text-muted);
		border-radius: var(--radius-md);
		transition:
			background-color 160ms ease-out,
			color 160ms ease-out;
	}

	a :global(svg) {
		flex: none;
	}

	a:hover {
		background: var(--accent-tint);
	}

	a[aria-current='page'] {
		background: var(--accent-tint);
		color: var(--accent);
	}

	a[aria-current='page']::after {
		content: '';
		position: absolute;
		inset-block: var(--space-2);
		inset-inline-end: calc(var(--space-5) * -1);
		inline-size: 3px;
		border-radius: var(--radius-full);
		background: var(--accent);
	}

	@media (min-width: 64rem) {
		nav {
			display: flex;
			flex-direction: column;
			gap: var(--space-1);
		}
	}
</style>
