<script lang="ts">
	import type { ComponentProps } from 'svelte';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import Icon from '#lib/components/Icon.svelte';
	import SegmentNav from '#lib/components/SegmentNav.svelte';
	import SeasonSwitch from '#lib/components/SeasonSwitch.svelte';
	import { GROUPS, GROUP_LABEL, type Group } from '#lib/site.js';
	import BackButton from './BackButton.svelte';
	import { shell } from './shell.svelte.js';

	type IconName = ComponentProps<typeof Icon>['name'];

	// 變體 C：頂部精簡列 + 底部分頁 + 兩塊 sheet。其他變體與桌面版隱藏。
	let {
		season,
		group,
		records = false,
		hrefFor
	}: {
		season?: string;
		group?: Group;
		records?: boolean;
		hrefFor: (param: 'season' | 'group', value: string) => string | undefined;
	} = $props();

	let ctxDlg = $state<HTMLDialogElement>();
	let moreDlg = $state<HTMLDialogElement>();
	let seasonDlg = $state<HTMLDialogElement>();

	const deep = $derived(!!(page.params.game || page.params.team || page.params.player));
	const base = $derived(season && group ? `${resolve('/[season]/[group]', { season, group })}/` : '');
	const recordsBase = $derived(group ? `${resolve('/records/[group]/[[kind=recordKind]]', { group })}/` : '');
	const onRecords = $derived(page.url.pathname.startsWith(recordsBase));
	const rest = $derived(page.url.pathname.slice(base.length));
	const inSection = (s: string) => !onRecords && (rest === `${s}/` || rest.startsWith(`${s}/`));

	const tabs = $derived<{ label: string; icon: IconName; href: string; current: boolean }[]>([
		{ label: '首頁', icon: 'home', href: base, current: !onRecords && rest === '' },
		{ label: '賽事', icon: 'calendar', href: `${base}matches/`, current: inSection('matches') },
		{ label: '戰績', icon: 'trophy', href: `${base}standings/`, current: inSection('standings') },
		{ label: '球隊', icon: 'shield', href: `${base}teams/`, current: inSection('teams') }
	]);
	const more = $derived<{ label: string; icon: IconName; href: string }[]>([
		{ label: '球員', icon: 'user', href: `${base}players/` },
		{ label: '排行榜', icon: 'podium', href: `${base}leaders/` },
		{ label: '歷史紀錄', icon: 'trophy', href: recordsBase }
	]);
	const moreCurrent = $derived(onRecords || inSection('players') || inSection('leaders'));
	const ctxLabel = $derived(group ? (records || !season ? GROUP_LABEL[group] : `${season}・${GROUP_LABEL[group]}`) : '');

	function closeOnLink(e: MouseEvent, dlg?: HTMLDialogElement) {
		if ((e.target as Element).closest('a')) dlg?.close();
	}
	function closeOnBackdrop(e: MouseEvent) {
		if (e.target === e.currentTarget) (e.currentTarget as HTMLDialogElement).close();
	}
</script>

{#if season && group}
	<div class="top">
		{#if deep}<BackButton />{:else}<a class="brand" href={resolve('/')}>TVL 數據</a>{/if}
		{#if shell.variant === 'D'}
			<div class="ctrls">
				<div class="seg">
					<SegmentNav
						label="組別"
						items={GROUPS.map((g) => ({ key: g, label: GROUP_LABEL[g].replace('組', ''), href: hrefFor('group', g), current: g === group }))}
					/>
				</div>
				{#if !records}
					<button type="button" class="ctx" aria-haspopup="dialog" aria-label="切換賽季，目前 {season}" onclick={() => seasonDlg?.showModal()}>
						<span>{season}</span><Icon name="chevron-down" size={18} />
					</button>
				{/if}
			</div>
		{:else}
			<button
				type="button"
				class="ctx"
				aria-haspopup="dialog"
				aria-label="切換賽季與組別，目前 {ctxLabel.replace('・', ' ')}"
				onclick={() => ctxDlg?.showModal()}
			>
				<span>{ctxLabel}</span><Icon name="chevron-down" size={18} />
			</button>
		{/if}
	</div>

	<nav class="tabs" aria-label="主要分頁">
		{#each tabs as t (t.label)}
			<a href={t.href} aria-current={t.current ? 'page' : undefined}>
				<Icon name={t.icon} size={24} /><span>{t.label}</span>
			</a>
		{/each}
		<button type="button" aria-current={moreCurrent ? 'page' : undefined} onclick={() => moreDlg?.showModal()}>
			<Icon name="more" size={24} /><span>更多</span>
		</button>
	</nav>

	<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
	<dialog class="sheet" bind:this={moreDlg} aria-label="更多" onclick={closeOnBackdrop}>
		<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
		<div class="inner" onclick={(e) => closeOnLink(e, moreDlg)}>
			<h2>更多</h2>
			{#each more as m (m.label)}
				<a class="row" href={m.href}><Icon name={m.icon} size={24} /><span>{m.label}</span></a>
			{/each}
		</div>
	</dialog>

	<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
	<dialog class="sheet" bind:this={seasonDlg} aria-label="賽季" onclick={closeOnBackdrop}>
		<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
		<div class="inner" onclick={(e) => closeOnLink(e, seasonDlg)}>
			<h2>賽季</h2>
			<SeasonSwitch current={season} hrefFor={(s) => hrefFor('season', s)} />
			<p class="note">非官方・資料整理自聯賽官網</p>
		</div>
	</dialog>

	<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
	<dialog class="sheet" bind:this={ctxDlg} aria-label="賽季與組別" onclick={closeOnBackdrop}>
		<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
		<div class="inner" onclick={(e) => closeOnLink(e, ctxDlg)}>
			<h2>賽季與組別</h2>
			<SegmentNav
				label="組別"
				items={GROUPS.map((g) => ({ key: g, label: GROUP_LABEL[g], href: hrefFor('group', g), current: g === group }))}
			/>
			{#if !records}
				<SeasonSwitch current={season} hrefFor={(s) => hrefFor('season', s)} />
			{/if}
			<p class="note">非官方・資料整理自聯賽官網</p>
		</div>
	</dialog>
{/if}

<style>
	.top,
	.tabs {
		display: none;
	}

	.sheet {
		position: fixed;
		inset: auto 0 0 0;
		margin: 0;
		padding: 0;
		inline-size: 100%;
		max-inline-size: none;
		max-block-size: 85dvh;
		border: 0;
		border-radius: var(--radius-lg) var(--radius-lg) 0 0;
		background: var(--surface);
		color: var(--text);
		box-shadow: var(--shadow-pop);
	}

	.sheet::backdrop {
		background: rgba(18, 20, 31, 0.5);
	}

	.inner {
		display: grid;
		gap: var(--space-3);
		padding: var(--space-4) calc(var(--space-4) + var(--safe-right)) calc(var(--space-4) + var(--safe-bottom))
			calc(var(--space-4) + var(--safe-left));
	}

	h2 {
		margin: 0;
		font-size: var(--text-lg);
	}

	.row {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		min-block-size: 48px;
		font-size: var(--text-md);
		font-weight: 700;
	}

	.note {
		margin: 0;
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	@media (max-width: 63.99rem) {
		:global(html:is([data-shell='C'], [data-shell='D'])) .top {
			display: flex;
			align-items: center;
			justify-content: space-between;
			gap: var(--space-3);
			position: sticky;
			inset-block-start: 0;
			z-index: 2;
			box-sizing: content-box;
			block-size: 44px;
			padding: var(--safe-top) var(--space-4) 0;
			background: var(--bg);
		}

		:global(html:is([data-shell='C'], [data-shell='D'])) .tabs {
			display: grid;
			grid-template-columns: repeat(5, 1fr);
			position: fixed;
			inset: auto 0 0 0;
			z-index: 5;
			box-sizing: content-box;
			block-size: var(--tabbar-h);
			padding: 0 var(--safe-right) var(--safe-bottom) var(--safe-left);
			background: var(--surface);
			border-block-start: 1px solid var(--line);
		}

		.brand {
			font-size: var(--text-lg);
			font-weight: 800;
		}

		.ctx {
			display: inline-flex;
			align-items: center;
			gap: var(--space-1);
			flex: none;
			white-space: nowrap;
			min-block-size: 44px;
			padding: 0 var(--space-3) 0 var(--space-4);
			border: 0;
			border-radius: var(--radius-full);
			background: var(--accent-tint);
			color: var(--accent);
			font: inherit;
			font-size: var(--text-md);
			font-weight: 700;
		}

		.ctrls {
			display: flex;
			align-items: center;
			flex-wrap: nowrap;
			gap: var(--space-2);
			min-inline-size: 0;
			margin-inline-start: auto;
		}

		.seg {
			flex: none;
		}

		.seg :global(nav) {
			inline-size: auto;
		}

		.seg :global(a) {
			padding-inline: var(--space-3);
		}

		@media (max-width: 22rem) {
			.brand {
				display: none;
			}

			:global(html:is([data-shell='C'], [data-shell='D'])) .top {
				padding-inline: var(--space-2);
				gap: var(--space-1);
			}

			.ctrls {
				gap: var(--space-1);
			}

			.seg :global(a) {
				padding-inline: var(--space-2);
			}

			.ctx {
				padding: 0 var(--space-2) 0 var(--space-3);
			}
		}

		.tabs a,
		.tabs button {
			display: flex;
			flex-direction: column;
			align-items: center;
			justify-content: center;
			gap: 2px;
			padding: 0;
			border: 0;
			background: transparent;
			font: inherit;
			font-size: var(--text-sm);
			font-weight: 700;
			color: var(--text-muted);
		}

		.tabs [aria-current='page'] {
			color: var(--accent);
		}
	}
</style>
