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
					<table class="rank-table" aria-label="{board.label}單場前三名">
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

	/* 表格本身的樣式是 app.css 的 .rank-table，這裡只放單場前三名自己的差異 */
	tbody th[scope='row'] {
		font-weight: 400;
		text-align: start;
	}

	td.num {
		font-size: var(--text-lg);
	}

	.identity {
		display: grid;
		min-inline-size: 0;
		flex: 1;
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
