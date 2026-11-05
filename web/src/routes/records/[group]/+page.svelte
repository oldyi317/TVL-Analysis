<script lang="ts">
	import EmptyState from '#lib/components/EmptyState.svelte';
	import RecordBoard from '#lib/components/RecordBoard.svelte';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { formatThroughDate } from '#lib/leaders.js';
	import { coverageText } from '#lib/records.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<SiteHead title="{GROUP_LABEL[data.group]} 歷史紀錄｜TVL 數據" />

<h1>{GROUP_LABEL[data.group]} 歷史紀錄</h1>

{#if !data.coverage}
	<EmptyState title="尚無單場數據" hint="單場數據匯入後就會出現" />
{:else}
	<p class="scope">
		{coverageText(data.coverage)} · 統計至 {formatThroughDate(data.coverage.through_date)}
	</p>
	<section aria-labelledby="season-records">
		<h2 id="season-records">單季紀錄</h2>
		<p class="intro">一名球員在一個賽季內的總和，同一人在不同賽季可各自上榜。</p>
		<div class="boards">
			{#each data.boards as board (board.key)}
				<RecordBoard {board} group={data.group} />
			{/each}
		</div>
	</section>
	<ul class="notes" aria-label="附註">
		<li>出賽場數以有技術統計的場次計：一場中任一項技術統計不為 0 才算一場。</li>
		<li>不含黃金決勝局。</li>
		<li>標示「進行中」的紀錄來自本季，隨賽事進行可能被刷新。</li>
	</ul>
{/if}

<style>
	h1 {
		margin-block: var(--space-2) var(--space-1);
	}

	.scope {
		margin: 0 0 var(--space-5);
		font-size: var(--text-sm);
		color: var(--text-muted);
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
