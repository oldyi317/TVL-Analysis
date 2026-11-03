<script lang="ts">
	import { resolve } from '$app/paths';
	import { visibleRows, type LeaderBoardView } from '#lib/leaders.js';
	import { positionLabel, type Group } from '#lib/site.js';
	import RankDot from './RankDot.svelte';
	import TeamBadge from './TeamBadge.svelte';

	let { board, season, group }: { board: LeaderBoardView; season: string; group: Group } = $props();

	const uid = $props.id();
	let open = $state(false);
	let section = $state<HTMLElement>();

	// 收合後把卡片頂端帶回視窗，讀者不會被丟在頁尾
	function toggle() {
		open = !open;
		if (!open) section?.scrollIntoView({ block: 'start' });
	}

	const top = $derived(visibleRows(board.rows));
	const rows = $derived(open ? board.rows : top);
	const expandable = $derived(board.rows.length > top.length);

	const playerHref = (playerId: number) =>
		`${resolve('/[season]/[group]/players/[player]', { season, group, player: String(playerId) })}/`;
</script>

<section aria-labelledby="{uid}-h" bind:this={section}>
	<h2 id="{uid}-h">{board.label}</h2>
	<div class="card">
		<table id="{uid}-t">
			<thead>
				<tr>
					<th scope="col" class="rank">名次</th>
					<th scope="col">球員</th>
					<th scope="col" class="num">{board.label}</th>
				</tr>
			</thead>
			<tbody>
				{#each rows as row (row.player_id)}
					<tr>
						<td class="rank"><RankDot rank={row.rank} /></td>
						<td>
							<span class="player">
								<TeamBadge team={row.team} size="sm" label={row.team.short_name} />
								<a href={playerHref(row.player_id)}>{row.name}</a>
								{#if row.position}
									<span class="position">{positionLabel(row.position)}</span>
								{/if}
							</span>
						</td>
						<td class="num">{row.value}</td>
					</tr>
				{/each}
			</tbody>
		</table>
		{#if expandable}
			<button type="button" aria-expanded={open} aria-controls="{uid}-t" onclick={toggle}>
				{open ? '收合' : `展開全部（${board.rows.length} 人）`}
			</button>
		{/if}
	</div>
</section>

<style>
	section {
		scroll-margin-block-start: var(--space-4);
	}

	h2 {
		margin-block: 0 var(--space-3);
	}

	.card {
		padding-block: var(--space-2);
		padding-inline: var(--space-3);
	}

	table {
		inline-size: 100%;
		border-collapse: collapse;
		font-size: var(--text-md);
	}

	th {
		padding: var(--space-2) var(--space-2);
		border-block-end: 1px solid var(--line);
		color: var(--text-muted);
		font-size: var(--text-sm);
		font-weight: 600;
		text-align: start;
		white-space: nowrap;
	}

	td {
		block-size: 3rem;
		padding: 0 var(--space-2);
		border-block-end: 1px solid var(--line);
	}

	tbody tr:last-child td {
		border-block-end: 0;
	}

	.rank {
		inline-size: 2.5rem;
		text-align: center;
	}

	.num {
		text-align: end;
	}

	td.num {
		font-weight: var(--weight-num);
		font-variant-numeric: tabular-nums;
	}

	.player {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		min-inline-size: 0;
	}

	.player a {
		min-inline-size: 0;
		font-weight: 700;
		color: var(--text);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.player a:hover {
		color: var(--accent);
		text-decoration: underline;
		text-underline-offset: 0.2em;
	}

	.position {
		flex: none;
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	button {
		display: flex;
		align-items: center;
		justify-content: center;
		inline-size: 100%;
		min-block-size: var(--tap);
		margin-block-start: var(--space-1);
		padding: 0;
		border: 0;
		border-block-start: 1px solid var(--line);
		border-radius: 0;
		background: none;
		color: var(--accent);
		font: inherit;
		font-size: var(--text-sm);
		font-weight: 700;
		cursor: pointer;
		transition: color 160ms ease-out;
	}

	button:hover {
		color: var(--accent-strong);
		text-decoration: underline;
		text-underline-offset: 0.2em;
	}

	@media (min-width: 64rem) {
		.card {
			padding-block: var(--space-3);
			padding-inline: var(--space-4);
		}
	}
</style>
