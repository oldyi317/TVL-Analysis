<script lang="ts">
	import type { ComponentProps } from 'svelte';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import type { Group } from '#lib/site.js';
	import Icon from './Icon.svelte';

	type IconName = ComponentProps<typeof Icon>['name'];

	let { season, group }: { season: string; group: Group } = $props();

	let nav = $state<HTMLElement>();

	// 手機版導覽可橫捲：選中項若超出右緣就捲到可見，讓讀者一眼看到「你在哪」
	$effect(() => {
		const current = nav?.querySelector<HTMLElement>('[aria-current="page"]');
		if (!nav || !current) return;
		const overflow = current.getBoundingClientRect().right - nav.getBoundingClientRect().right;
		if (overflow > 0) nav.scrollLeft += overflow;
	});

	const base = $derived(`${resolve('/[season]/[group]', { season, group })}/`);
	const recordsBase = $derived(`${resolve('/records/[group]/[[kind=recordKind]]', { group })}/`);
	const onRecords = $derived(page.url.pathname.startsWith(recordsBase));
	const items = $derived<{ label: string; icon: IconName; href: string; section: string }[]>([
		{ label: '首頁', icon: 'home', href: base, section: '' },
		{ label: '賽事', icon: 'calendar', href: `${base}matches/`, section: 'matches' },
		{ label: '戰績', icon: 'trophy', href: `${base}standings/`, section: 'standings' },
		{ label: '球隊', icon: 'shield', href: `${base}teams/`, section: 'teams' },
		{ label: '球員', icon: 'user', href: `${base}players/`, section: 'players' },
		{ label: '排行榜', icon: 'podium', href: `${base}leaders/`, section: 'leaders' },
		{ label: '歷史紀錄', icon: 'trophy', href: recordsBase, section: 'records' }
	]);

	function isCurrent(section: string): boolean {
		if (section === 'records') return onRecords;
		if (onRecords) return false;
		const rest = page.url.pathname.slice(base.length);
		return section === '' ? rest === '' : rest === `${section}/` || rest.startsWith(`${section}/`);
	}
</script>

<nav aria-label="頁面" bind:this={nav}>
	{#each items as item (item.label)}
		<a href={item.href} aria-current={isCurrent(item.section) ? 'page' : undefined}>
			<Icon name={item.icon} />
			<span>{item.label}</span>
		</a>
	{/each}
</nav>

<style>
	nav {
		display: flex;
		gap: var(--space-1);
		overflow-x: auto;
		scroll-padding-inline: var(--space-3);
		scrollbar-width: none;
	}

	a:last-child {
		scroll-margin-inline-end: var(--space-3);
	}

	nav::-webkit-scrollbar {
		display: none;
	}

	a {
		position: relative;
		flex: 1 0 auto;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-3);
		min-block-size: var(--tap);
		padding-inline: var(--space-3);
		font-size: var(--text-md);
		font-weight: 700;
		color: var(--text-muted);
		border-radius: var(--radius-full);
		transition:
			background-color 160ms ease-out,
			color 160ms ease-out;
	}

	a :global(svg) {
		display: none;
	}

	a:hover {
		background: var(--accent-tint);
	}

	a[aria-current='page'] {
		background: var(--accent);
		color: var(--on-accent);
	}

	@media (min-width: 64rem) {
		nav {
			flex-direction: column;
			overflow-x: visible;
		}

		a {
			flex: none;
			justify-content: flex-start;
			min-block-size: 2.75rem;
			padding-inline: var(--space-3);
			border-radius: var(--radius-md);
		}

		a :global(svg) {
			display: block;
			flex: none;
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
	}
</style>
