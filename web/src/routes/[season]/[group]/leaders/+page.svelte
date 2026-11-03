<script lang="ts">
	import EmptyState from '#lib/components/EmptyState.svelte';
	import LeaderBoard from '#lib/components/LeaderBoard.svelte';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { formatThroughDate } from '#lib/leaders.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<SiteHead title="{data.season.name} {GROUP_LABEL[data.group]} 排行榜｜TVL 數據" />

<h1>{data.season.name} {GROUP_LABEL[data.group]} 排行榜</h1>

{#if data.boards.length === 0}
	<EmptyState title="本季尚無單場數據" hint="單場數據匯入後就會出現" />
{:else}
	{#if data.summary?.through_date}
		<p class="scope">統計至 {formatThroughDate(data.summary.through_date)} · 共 {data.summary.games} 場</p>
	{/if}
	<div class="boards">
		{#each data.boards as board, i (board.key)}
			<div class="rise" style="--i:{Math.min(i + 1, 6)}">
				<LeaderBoard {board} season={data.season.name} group={data.group} />
			</div>
		{/each}
	</div>
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

	.boards {
		display: grid;
		gap: var(--space-5);
		align-items: start;
	}

	@media (min-width: 40rem) {
		.boards {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}

	@media (min-width: 64rem) {
		.boards {
			grid-template-columns: repeat(3, minmax(0, 1fr));
		}
	}
</style>
