<script lang="ts">
	import { resolve } from '$app/paths';
	import { GROUPS, GROUP_SHORT_LABEL, type Group } from '#lib/site.js';
	import BottomSheet from './BottomSheet.svelte';
	import Icon from './Icon.svelte';
	import SegmentNav from './SegmentNav.svelte';
	import SeasonSwitch from './SeasonSwitch.svelte';

	let {
		season,
		group,
		records = false,
		back,
		hrefFor
	}: {
		season?: string;
		group?: Group;
		records?: boolean;
		back?: { href: string; canGoBack: boolean };
		hrefFor: (param: 'season' | 'group', value: string) => string | undefined;
	} = $props();

	let sheet = $state<BottomSheet>();
</script>

<header>
	{#if back}
		<a
			class="back"
			href={back.href}
			aria-label="返回"
			onclick={(event) => {
				if (!back.canGoBack) return;
				if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
				event.preventDefault();
				history.back();
			}}
		>
			<Icon name="chevron-left" size={24} />
		</a>
	{:else}
		<a class="brand" href={resolve('/')}>TVL 數據</a>
	{/if}
	{#if season && group}
		<div class="controls">
			<div class="groups">
				<SegmentNav
					label="組別"
					items={GROUPS.map((g) => ({ key: g, label: GROUP_SHORT_LABEL[g], href: hrefFor('group', g), current: g === group }))}
				/>
			</div>
			{#if !records}
				<button
					type="button"
					class="chip season"
					aria-haspopup="dialog"
					aria-label="切換賽季，目前 {season}"
					onclick={(event) => sheet?.open(event.currentTarget)}
				>
					<span>{season}</span>
					<Icon name="chevron-down" size={16} />
				</button>
				<BottomSheet bind:this={sheet} label="賽季">
					<SeasonSwitch current={season} hrefFor={(s) => hrefFor('season', s)} />
				</BottomSheet>
			{/if}
		</div>
	{/if}
</header>

<style>
	header {
		position: sticky;
		inset-block-start: 0;
		z-index: var(--z-topbar);
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-2);
		block-size: var(--shell-top);
		padding-block-start: var(--safe-top);
		padding-inline: calc(var(--space-3) + var(--safe-left)) calc(var(--space-3) + var(--safe-right));
		background: var(--surface);
		border-block-end: 1px solid var(--card-edge);
		box-shadow: var(--shadow-card);
	}

	.brand {
		flex: none;
		font-size: var(--text-lg);
		font-weight: 800;
		color: var(--text);
		white-space: nowrap;
	}

	.back {
		flex: none;
		display: grid;
		place-items: center;
		inline-size: var(--tap);
		block-size: var(--tap);
		border-radius: var(--radius-full);
		color: var(--text);
	}

	.back:active {
		background: var(--accent-tint);
	}

	.controls {
		display: flex;
		align-items: center;
		gap: var(--space-1);
		min-inline-size: 0;
	}

	.groups :global(nav) {
		padding: 0;
		background: none;
	}

	.groups :global(a) {
		min-inline-size: var(--tap);
		min-block-size: var(--tap);
		padding-inline: var(--space-2);
		font-size: var(--text-sm);
		white-space: nowrap;
	}

	.season {
		gap: var(--space-1);
		padding-inline: var(--space-2);
		background: var(--accent-tint);
		color: var(--accent);
		font-weight: 700;
		white-space: nowrap;
	}

	@media (min-width: 64rem) {
		header {
			display: none;
		}
	}
</style>
