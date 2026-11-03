<script lang="ts">
	import EmptyState from '#lib/components/EmptyState.svelte';
	import LeaderBoard from '#lib/components/LeaderBoard.svelte';
	import SeasonChampions from '#lib/components/SeasonChampions.svelte';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { formatThroughDate } from '#lib/leaders.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	let current = $state<string>();
	let sections = $state<HTMLElement[]>([]);
	function track() {
		if (!sections.length) return;
		if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 1) {
			current = data.boards.at(-1)?.key;
			return;
		}
		const threshold = parseFloat(getComputedStyle(sections[0]).scrollMarginBlockStart) + 1;
		// 同列多卡以第一張為預設；若讀者已點同列項目，保留該選擇。
		let rowTop = -Infinity;
		let active: number[] = [];
		sections.forEach((section, i) => {
			const top = section.getBoundingClientRect().top;
			if (top > threshold) return;
			if (top > rowTop + 1) { rowTop = top; active = [i]; }
			else if (Math.abs(top - rowTop) <= 1) active.push(i);
		});
		if (!active.some((i) => data.boards[i]?.key === current)) {
			current = data.boards[active[0] ?? 0]?.key;
		}
	}
	$effect(() => {
		let frame = 0;
		const onScroll = () => {
			if (frame) return;
			frame = requestAnimationFrame(() => { track(); frame = 0; });
		};
		track();
		window.addEventListener('scroll', onScroll, { passive: true });
		window.addEventListener('resize', onScroll);
		return () => {
			window.removeEventListener('scroll', onScroll);
			window.removeEventListener('resize', onScroll);
			cancelAnimationFrame(frame);
		};
	});
</script>

<SiteHead title="{data.season.name} {GROUP_LABEL[data.group]} 排行榜｜TVL 數據" />

<h1>{data.season.name} {GROUP_LABEL[data.group]} 排行榜</h1>

{#if data.boards.length === 0}
	<EmptyState title="本季尚無單場數據" hint="單場數據匯入後就會出現" />
{:else}
	{#if data.summary?.through_date}
		<p class="scope">統計至 {formatThroughDate(data.summary.through_date)} · 共 {data.summary.games} 場</p>
	{/if}
	<SeasonChampions boards={data.boards} season={data.season.name} group={data.group} />
	<nav class="items" aria-label="計數榜項目">
		{#each data.boards as board (board.key)}
			<a class="chip" href="#b-{board.key}" aria-current={(current ?? data.boards[0]?.key) === board.key ? 'location' : undefined} onclick={() => (current = board.key)}>{board.label}</a>
		{/each}
	</nav>
	<div class="boards">
		{#each data.boards as board, i (board.key)}
			<div class="board-anchor" bind:this={sections[i]}>
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

	.items {
		position: sticky;
		inset-block-start: 0;
		z-index: 1;
		display: flex;
		gap: var(--space-1);
		padding-block: var(--space-2);
		margin-block-end: var(--space-4);
		overflow-x: auto;
		background: var(--bg);
	}
	.items a { flex: none; }
	.board-anchor { min-inline-size: 0; scroll-margin-block-start: calc(var(--tap) + 2 * var(--space-2) + var(--space-4)); }

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
