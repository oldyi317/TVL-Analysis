<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import type { Group } from '#lib/site.js';

	let { season, group }: { season: string; group: Group } = $props();

	const base = $derived(`${resolve('/[season]/[group]', { season, group })}/`);
	const items = $derived([
		{ label: '首頁', href: base, section: '' },
		{ label: '賽事', href: `${base}matches/`, section: 'matches' },
		{ label: '戰績', href: `${base}standings/`, section: 'standings' },
		{ label: '球隊', href: `${base}teams/`, section: 'teams' },
		{ label: '球員', href: `${base}teams/`, section: 'players' }
	]);

	function isCurrent(section: string): boolean {
		if (section === 'players') return false;
		const rest = page.url.pathname.slice(base.length);
		return section === '' ? rest === '' : rest === `${section}/` || rest.startsWith(`${section}/`);
	}
</script>

<nav aria-label="頁面">
	{#each items as item (item.label)}
		<a href={item.href} aria-current={isCurrent(item.section) ? 'page' : undefined}>{item.label}</a>
	{/each}
</nav>

<style>
	nav {
		display: flex;
		gap: var(--space-1);
		overflow-x: auto;
		scrollbar-width: none;
	}

	nav::-webkit-scrollbar {
		display: none;
	}

	a {
		flex: 1 0 auto;
		display: flex;
		align-items: center;
		justify-content: center;
		min-block-size: var(--tap);
		padding-inline: var(--space-3);
		font-size: var(--text-md);
		font-weight: 700;
		color: var(--text-muted);
		border-radius: var(--radius-sm);
	}

	a[aria-current='page'] {
		background: var(--accent);
		color: var(--on-accent);
	}
</style>
