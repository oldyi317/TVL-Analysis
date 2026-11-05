<script lang="ts">
	import { resolve } from '$app/paths';
	import { isMatch, liveLabel, matchLabel, rowDetail, rowKey, type RecordRowView } from '#lib/records.js';
	import type { Group } from '#lib/site.js';
	import EmptyState from './EmptyState.svelte';
	import RankDot from './RankDot.svelte';
	import TeamBadge from './TeamBadge.svelte';

	let {
		board,
		rows,
		kindLabel,
		group
	}: { board: { key: string; label: string }; rows: RecordRowView[]; kindLabel: string; group: Group } = $props();

	const uid = $props.id();

	const playerHref = (season: string, playerId: number) =>
		`${resolve('/[season]/[group]/players/[player]', { season, group, player: String(playerId) })}/`;
	const matchHref = (season: string, gameId: number) =>
		`${resolve('/[season]/[group]/matches/[game]', { season, group, game: String(gameId) })}/`;
</script>

<section id="r-{board.key}" aria-labelledby="{uid}-h">
	<h3 id="{uid}-h">{board.label}</h3>
	<div class="card">
		{#if rows.length === 0}
			<EmptyState title="尚無紀錄" />
		{:else}
			<table aria-label="{board.label}{kindLabel}">
				<thead>
					<tr>
						<th scope="col" class="rank">名次</th>
						<th scope="col">球員</th>
						<th scope="col" class="num">{board.label}</th>
					</tr>
				</thead>
				<tbody>
					{#each rows as row (rowKey(row))}
						<tr>
							<td class="rank"><RankDot rank={row.rank} /></td>
							<td>
								<div class="player">
									<TeamBadge team={row.team} size="sm" label={row.team_short} />
									<div class="identity">
										<a class="name" href={playerHref(row.season, row.player_id)}>{row.name}</a>
										<span class="detail">{rowDetail(row)}{#if isMatch(row)}
												·
												{#if row.game_id !== null}<a class="match" href={matchHref(row.season, row.game_id)}
														>{matchLabel(row)}</a
													>{:else}{matchLabel(row)}{/if}{/if}{#if row.in_progress}<span class="live">
													· {liveLabel(row)}</span
												>{/if}</span
										>
									</div>
								</div>
							</td>
							<td class="num">{row.value}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</div>
</section>

<style>
	h3 {
		margin-block: 0 var(--space-3);
	}

	.card {
		padding-block: var(--space-2);
		padding-inline: var(--space-3);
	}

	table {
		inline-size: 100%;
		table-layout: fixed;
		border-collapse: collapse;
		font-size: var(--text-md);
	}

	th {
		padding: var(--space-2);
		border-block-end: 1px solid var(--line);
		color: var(--text-muted);
		font-size: var(--text-sm);
		font-weight: 600;
		text-align: start;
		white-space: nowrap;
	}

	td {
		padding: var(--space-2);
		border-block-end: 1px solid var(--line);
		vertical-align: middle;
	}

	tbody tr:last-child td {
		border-block-end: 0;
	}

	.rank {
		inline-size: 2.5rem;
		text-align: center;
	}

	.num {
		inline-size: 4.5rem;
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

	.identity {
		display: grid;
		min-inline-size: 0;
		flex: 1;
	}

	.name {
		min-block-size: var(--tap);
		display: flex;
		align-items: center;
		font-weight: 700;
		color: var(--text);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.name:hover {
		color: var(--accent);
		text-decoration: underline;
		text-underline-offset: 0.2em;
	}

	.detail {
		font-size: var(--text-sm);
		color: var(--text-muted);
		line-height: 1.4;
		overflow-wrap: anywhere;
	}

	.match {
		color: inherit;
		text-decoration: underline;
		text-underline-offset: 0.2em;
	}

	.match:hover {
		color: var(--accent);
	}

	.live {
		font-weight: 700;
		color: var(--accent);
	}

	@media (min-width: 64rem) {
		.card {
			padding-block: var(--space-3);
			padding-inline: var(--space-4);
		}
	}
</style>
