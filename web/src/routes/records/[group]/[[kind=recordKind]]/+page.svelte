<script lang="ts">
	import { resolve } from '$app/paths';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import RecordBoard from '#lib/components/RecordBoard.svelte';
	import SegmentNav from '#lib/components/SegmentNav.svelte';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { formatThroughDate } from '#lib/leaders.js';
	import { coverageText, RECORD_KINDS, rowsOf, type RecordKindInfo } from '#lib/records.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const kindHref = (path: RecordKindInfo['path']) =>
		`${resolve('/records/[group]/[[kind=recordKind]]', { group: data.group, kind: path || undefined })}/`;
</script>

<SiteHead title="{GROUP_LABEL[data.group]} {data.kind.label}｜歷史紀錄｜TVL 數據" />

<h1>{GROUP_LABEL[data.group]} 歷史紀錄</h1>

{#if !data.coverage}
	<EmptyState title="尚無單場數據" hint="單場數據匯入後就會出現" />
{:else}
	<p class="scope">
		{coverageText(data.coverage)} · 統計至 {formatThroughDate(data.coverage.through_date)}
	</p>
	<div class="kinds">
		<SegmentNav
			label="紀錄類型"
			items={RECORD_KINDS.map((k) => ({ key: k.key, label: k.label, href: kindHref(k.path), current: k.key === data.kind.key }))}
			stay
		/>
	</div>
	<section aria-labelledby="records-kind">
		<h2 id="records-kind">{data.kind.label}</h2>
		<p class="intro">{data.kind.intro}</p>
		<div class="boards">
			{#each data.boards as board (board.key)}
				<RecordBoard
					{board}
					rows={rowsOf(board, data.kind.key)}
					kindLabel={data.kind.label}
					group={data.group}
				/>
			{/each}
		</div>
	</section>
	<ul class="notes" aria-label="附註">
		<li>出賽場數以有技術統計的場次計：一場中任一項技術統計不為 0 才算一場。</li>
		<li>不含黃金決勝局。</li>
		<li>標示「進行中」的單季紀錄與單場紀錄來自本季，標示「含進行中賽季」的累計紀錄計入本季，隨賽事進行可能被刷新。</li>
	</ul>
{/if}

<style>
	h1 {
		margin-block: var(--space-2) var(--space-1);
	}

	.scope {
		margin: 0 0 var(--space-4);
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.kinds {
		margin-block-end: var(--space-5);
	}

	h2 {
		margin-block: 0 var(--space-1);
	}

	.intro {
		margin: 0 0 var(--space-4);
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.boards {
		display: grid;
		gap: var(--space-5);
		align-items: start;
		margin-block-end: var(--space-5);
	}

	.notes {
		margin: 0;
		padding-inline-start: var(--space-5);
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	@media (min-width: 40rem) {
		.boards {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
</style>
