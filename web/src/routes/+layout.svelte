<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { asset, resolve } from '$app/paths';
	import { GROUPS, GROUP_LABEL, seasons } from '#lib/site.js';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();

	// 切換賽季或組別：只替換路徑中該片段的第一次出現，停留在同一頁面類型。
	// base 是固定的 /TVL-Analysis，賽季與組別是 base 之後的前兩段，不會與其他片段撞名。
	function switchHref(param: 'season' | 'group', value: string): string | undefined {
		const current = page.params[param];
		if (!current) return undefined;
		return page.url.pathname.replace(`/${current}/`, `/${value}/`);
	}
</script>

<svelte:head>
	<link rel="icon" href={asset('favicon.svg')} />
</svelte:head>

<header class="site">
	<div class="wrap">
		<a class="brand" href={resolve('/')}>TVL 數據</a>
		<span class="note">非官方・資料整理自聯賽官網</span>
	</div>
	{#if page.params.season && page.params.group}
		<nav class="wrap seasons" aria-label="賽季">
			{#each seasons as season (season.cup_id)}
				<a
					href={switchHref('season', season.name)}
					aria-current={page.params.season === season.name ? 'page' : undefined}
				>
					{season.name}
				</a>
			{/each}
		</nav>
		<nav class="wrap groups" aria-label="組別">
			{#each GROUPS as group (group)}
				<a
					href={switchHref('group', group)}
					aria-current={page.params.group === group ? 'page' : undefined}
				>
					{GROUP_LABEL[group]}
				</a>
			{/each}
		</nav>
	{/if}
</header>

<main class="wrap">
	{@render children()}
</main>

<style>
	.wrap {
		max-inline-size: var(--measure);
		margin-inline: auto;
		padding-inline: 1rem;
	}

	.site {
		background: var(--surface);
		border-bottom: 1px solid var(--line);
	}

	.site > .wrap:first-child {
		display: flex;
		align-items: baseline;
		gap: 0.75rem;
		padding-block-start: 0.25rem;
	}

	.brand {
		display: inline-flex;
		align-items: center;
		min-block-size: var(--tap);
		font-size: 1.125rem;
		font-weight: 600;
	}

	.note {
		font-size: 0.75rem;
		color: var(--text-muted);
	}

	nav {
		display: flex;
		gap: 0.25rem;
	}

	/* 第一個連結不留左內距，文字與底線都從內容左緣起算 */
	nav a:first-child {
		padding-inline-start: 0;
	}

	nav a {
		display: inline-flex;
		align-items: center;
		min-block-size: var(--tap);
		padding-inline: 0.75rem;
		color: var(--text-muted);
		border-bottom: 2px solid transparent;
	}

	nav a:hover,
	nav a:active {
		color: var(--text);
	}

	nav a[aria-current='page'] {
		color: var(--text);
		border-bottom-color: var(--text);
	}

	.seasons a {
		font-size: 0.875rem;
	}

	.groups {
		font-weight: 600;
	}

	main {
		padding-block-end: 3rem;
	}
</style>
