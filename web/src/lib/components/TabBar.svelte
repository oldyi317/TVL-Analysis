<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { DEFAULT_GROUP } from '#lib/groupMemory.js';
	import { inPhoneBrowserTab } from '#lib/installGuide.js';
	import { installPrompt } from '#lib/installPrompt.svelte.js';
	import { hrefOf, MORE_SECTIONS, NAV_ITEMS, sectionOf, tabOf, type Section } from '#lib/nav.js';
	import siteJson from '#lib/data/site.json';
	import { formatTaipeiMonthDay } from '#lib/datetime.js';
	import { connection } from '#lib/serviceWorker.svelte.js';
	import { currentSeason, type Group } from '#lib/site.js';
	import BottomSheet from './BottomSheet.svelte';
	import Icon from './Icon.svelte';

	let { season, group }: { season?: string; group?: Group } = $props();

	const base = $derived(`${resolve('/[season]/[group]', { season: encodeURIComponent(season ?? currentSeason.name), group: group ?? DEFAULT_GROUP })}/`);
	const recordsBase = $derived(`${resolve('/records/[group]/[[kind=recordKind]]', { group: group ?? DEFAULT_GROUP })}/`);
	const href = (section: Section) => hrefOf(section, base, recordsBase);
	const notificationsHref = `${resolve('/notifications')}/`;

	const tab = $derived(tabOf(page.route.id));
	const section = $derived(sectionOf(page.route.id));
	const tabItems = NAV_ITEMS.filter((item) => !MORE_SECTIONS.includes(item.section));
	const moreItems = NAV_ITEMS.filter((item) => MORE_SECTIONS.includes(item.section));

	const updatedAt = formatTaipeiMonthDay(siteJson.generated_at);

	let sheet: BottomSheet;
</script>

<nav aria-label="主要分頁">
	{#each tabItems as item (item.section)}
		<a href={href(item.section)} aria-current={tab === item.section ? 'page' : undefined}>
			<Icon name={item.icon} />
			<span>{item.label}</span>
		</a>
	{/each}
	<button
		type="button"
		aria-haspopup="dialog"
		aria-current={tab === 'more' ? 'page' : undefined}
		onclick={(event) => sheet.open(event.currentTarget)}
	>
		<Icon name="more" />
		<span>更多</span>
	</button>
</nav>

<BottomSheet bind:this={sheet} label="更多">
	<ul>
		{#each moreItems as item (item.section)}
			<li>
				<a href={href(item.section)} aria-current={section === item.section ? 'page' : undefined}>
					<Icon name={item.icon} />
					<span>{item.label}</span>
				</a>
			</li>
		{/each}
		<li>
			<a href={notificationsHref} aria-current={section === 'notifications' ? 'page' : undefined}>
				<Icon name="bell" />
				<span>通知</span>
			</a>
		</li>
		{#if installPrompt.guide && inPhoneBrowserTab(installPrompt.guide)}
			<li>
				<a href={notificationsHref}>
					<Icon name="add-square" />
					<span>加到主畫面</span>
				</a>
			</li>
		{/if}
	</ul>
	{#if connection.status === 'ready'}
		<p class="offline-status">可離線使用，資料更新至 {updatedAt}</p>
	{:else if connection.status === 'preparing'}
		<p class="offline-status">離線資料準備中</p>
	{/if}
</BottomSheet>

<style>
	nav {
		position: fixed;
		inset-inline: 0;
		inset-block-end: 0;
		z-index: var(--z-tabbar);
		display: flex;
		block-size: calc(var(--tabbar-h) + var(--safe-bottom));
		padding-inline: var(--safe-left) var(--safe-right);
		padding-block-end: var(--safe-bottom);
		background: var(--surface);
		border-block-start: 1px solid var(--line);
	}

	nav a,
	nav button {
		position: relative;
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--space-1);
		min-inline-size: var(--tap);
		min-block-size: var(--tap);
		padding: 0;
		border: 0;
		background: none;
		font: inherit;
		font-size: var(--text-sm);
		line-height: 1.2;
		color: var(--text-muted);
		cursor: pointer;
	}

	nav a:focus-visible,
	nav button:focus-visible {
		outline-offset: calc(var(--space-1) * -1);
	}

	nav [aria-current='page'] {
		color: var(--accent);
		font-weight: 700;
	}

	nav [aria-current='page']::before {
		content: '';
		position: absolute;
		inset-block-start: 0;
		inline-size: var(--space-6);
		block-size: var(--space-1);
		border-end-start-radius: var(--radius-full);
		border-end-end-radius: var(--radius-full);
		background: var(--accent);
	}

	ul {
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.offline-status {
		margin: var(--space-3) 0 0;
		padding-inline: var(--space-3);
		color: var(--text-muted);
		font-size: var(--text-sm);
	}

	li a {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		min-block-size: var(--tap);
		padding-inline: var(--space-3);
		font-size: var(--text-md);
		font-weight: 700;
		border-radius: var(--radius-md);
	}

	li a:hover {
		background: var(--accent-tint);
	}

	li a[aria-current='page'] {
		background: var(--accent-tint);
		color: var(--accent);
	}

	@media (min-width: 64rem) {
		nav {
			display: none;
		}
	}
</style>
