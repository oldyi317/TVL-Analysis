<script lang="ts">
	import { resolve } from '$app/paths';
	import type { MatchTopBoardView, MatchTopRowView } from '#lib/boxScore.js';
	import type { Group } from '#lib/site.js';
	import RankDot from './RankDot.svelte';
	import TeamBadge from './TeamBadge.svelte';

	let {
		boards,
		season,
		group
	}: {
		boards: readonly MatchTopBoardView[];
		season: string;
		group: Group;
	} = $props();

	const uid = $props.id();
	const visibleBoards = $derived(boards.filter((board) => board.rows.length > 0));

	const playerHref = (playerId: number) =>
		`${resolve('/[season]/[group]/players/[player]', { season, group, player: String(playerId) })}/`;
</script>

{#snippet playerInfo(row: MatchTopRowView)}
	<TeamBadge team={row.team} size="sm" />
	<span class="identity">
		<span class="name" title={row.name}>{row.name}</span>
		<span class="team">{row.team.short_name}</span>
	</span>
{/snippet}

{#if visibleBoards.length > 0}
	<div class="boards">
		{#each visibleBoards as board (board.key)}
			<section aria-labelledby="{uid}-{board.key}">
				<h3 id="{uid}-{board.key}">{board.label}</h3>
				<div class="card">
					<table aria-label="{board.label}單場前三名">
						<thead>
							<tr>
								<th scope="col" class="rank">名次</th>
								<th scope="col">球員</th>
								<th scope="col" class="num">{board.label}</th>
							</tr>
						</thead>
						<tbody>
							{#each board.rows as row (`${row.team.team_id}-${row.player_id}`)}
								<tr>
									<td class="rank"><RankDot rank={row.rank} /></td>
									<th scope="row">
										{#if row.linked}
											<a class="player" href={playerHref(row.player_id)}>
												{@render playerInfo(row)}
											</a>
										{:else}
											<div class="player">
												{@render playerInfo(row)}
											</div>
										{/if}
									</th>
									<td class="num">{row.value}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</section>
		{/each}
	</div>
{/if}

<style>
	.boards {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--space-5);
		align-items: start;
	}

	@media (min-width: 40rem) {
		.boards {
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: var(--space-5) var(--space-4);
		}
	}

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

	thead th {
		padding: var(--space-2);
		border-block-end: 1px solid var(--line);
		color: var(--text-muted);
		font-size: var(--text-sm);
		font-weight: 600;
		text-align: start;
		white-space: nowrap;
	}

	tbody th,
	tbody td {
		block-size: 3rem;
		padding: var(--space-1) var(--space-2);
		border-block-end: 1px solid var(--line);
		vertical-align: middle;
	}

	tbody tr:last-child th,
	tbody tr:last-child td {
		border-block-end: 0;
	}

	tbody th[scope='row'] {
		font-weight: 400;
		text-align: start;
	}

	.rank {
		inline-size: 2.5rem;
		text-align: center;
	}

	.num {
		inline-size: 5rem;
		text-align: end;
		white-space: nowrap;
	}

	td.num {
		font-size: var(--text-lg);
		font-weight: var(--weight-num);
		font-variant-numeric: tabular-nums;
		color: var(--text);
	}

	.player {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		min-block-size: var(--tap);
		min-inline-size: 0;
		color: var(--text);
	}

	.identity {
		display: grid;
		min-inline-size: 0;
		flex: 1;
	}

	.name {
		line-height: 1.35;
		font-weight: 700;
		color: var(--text);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	a.player:hover .name {
		color: var(--accent);
		text-decoration: underline;
		text-underline-offset: 0.2em;
	}

	.team {
		line-height: 1.35;
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--text-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	@media (min-width: 64rem) {
		.card {
			padding-block: var(--space-3);
			padding-inline: var(--space-4);
		}
	}
</style>
