<script lang="ts">
	import { resolve } from '$app/paths';
	import type { LeaderBoardView } from '#lib/leaders.js';
	import type { Group } from '#lib/site.js';

	let { boards, season, group }: { boards: LeaderBoardView[]; season: string; group: Group } = $props();
	const champions = $derived(boards.map((board) => ({
		...board, winners: board.rows.filter((row) => row.rank === 1 && row.title)
	})).filter((board) => board.winners.length));
	const href = (id: number) => `${resolve('/[season]/[group]/players/[player]', { season, group, player: String(id) })}/`;
</script>

{#if champions.length}
	<section aria-label="本季四王" class="champions rise" style="--i:1">
		<h2>本季四王</h2>
		<div class="card kings">
			{#each champions as board (board.key)}
				<div class="king" class:tied={board.winners.length > 1}>
					<span class="tag tag-glory">{board.winners[0].title}</span>
					<div class="names">
						{#each board.winners as winner (winner.player_id)}
							<a href={href(winner.player_id)} title="{winner.name} · {winner.team.short_name}">
								<strong>{winner.name}</strong><span class="team">{winner.team.short_name}</span>
							</a>
						{/each}
					</div>
					<div class="value"><strong>{board.winners[0].value}</strong><span>{board.label}</span></div>
				</div>
			{/each}
		</div>
	</section>
{/if}

<style>
	h2 { margin-block: 0 var(--space-3); }
	.champions { margin-block-end: var(--space-5); }
	.kings { padding: 0 var(--space-3); }
	.king { position: relative; display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: var(--space-2); align-items: center; min-block-size: 3.5rem; border-block-end: 1px solid var(--line); }
	.king:last-child { border: 0; }
	.names { display: flex; min-inline-size: 0; gap: var(--space-2); }
	a { display: flex; flex: 1; flex-wrap: wrap; align-content: center; align-items: baseline; column-gap: var(--space-1); min-inline-size: 0; min-block-size: 3.5rem; color: var(--text); }
	a strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-inline-size: 0; }
	/* 單一得主整列可點；並列時保留各自的入口。 */
	.king:not(.tied) a::after { content: ''; position: absolute; inset: 0; }
	a:hover strong { color: var(--accent); text-decoration: underline; }
	.team, .value span { font-size: var(--text-sm); color: var(--text-muted); }
	.value { display: flex; align-items: baseline; gap: var(--space-1); white-space: nowrap; }
	.value strong { font-size: var(--num-md); font-weight: var(--weight-num); font-variant-numeric: tabular-nums; }
	@media (min-width: 64rem) {
		.kings { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); }
		.king { padding: var(--space-3); border-block-end: 0; border-inline-end: 1px solid var(--line); grid-template-columns: auto minmax(0, 1fr); }
		.value { grid-column: 1 / -1; justify-content: end; }
	}
</style>
