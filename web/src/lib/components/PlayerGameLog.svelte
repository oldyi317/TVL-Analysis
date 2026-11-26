<script module lang="ts">
	import type { GameStats } from '#lib/advanced.js';

	const pair = (good: number, total: number) => total === 0 ? '—' : `${good}／${total}`;
	const details = [
		{ label: '攻擊次數', value: (s: GameStats) => s.attack_total },
		{ label: '發球次數', value: (s: GameStats) => s.serve_total },
		{ label: '接發球', value: (s: GameStats) => pair(s.receive_excellent, s.receive_total) },
		{ label: '防守', value: (s: GameStats) => pair(s.dig_excellent, s.dig_total) },
		{ label: '舉球', value: (s: GameStats) => pair(s.set_excellent, s.set_total) }
	];
</script>

<script lang="ts">
	import { resolve } from '$app/paths';
	import { playerGameLogRows, type PlayerGame } from '#lib/advanced.js';
	import type { Group } from '#lib/site.js';
	import Icon from './Icon.svelte';

	let { gameLog, season, group }: { gameLog: PlayerGame[]; season: string; group: Group } = $props();
	const rows = $derived(playerGameLogRows(gameLog));
	const uid = $props.id();
	let open = $state<Record<number, boolean>>({});
</script>

{#if rows.length}
	<section aria-labelledby="{uid}-title">
		<h2 id="{uid}-title">逐場紀錄</h2>
		<div class="card log-card">
			<table aria-labelledby="{uid}-title" aria-describedby="{uid}-note">
				<thead>
					<tr>
						<th scope="col">日期</th>
						<th scope="col">對手</th>
						<th scope="col">勝敗</th>
						<th scope="col" class="num">得分</th>
						<th scope="col" class="num">攻擊</th>
						<th scope="col" class="num">攔網</th>
						<th scope="col" class="num">發球</th>
						{#each details as col (col.label)}
							<th scope="col" class="num wide">{col.label}</th>
						{/each}
						<th scope="col" class="toggle"><span class="sr">詳細</span></th>
					</tr>
				</thead>
				<tbody>
					{#each rows as game (game.gameId)}
						{@const id = `${uid}-${game.gameId}`}
						<tr class="main">
							<th scope="row">
								<a
									href="{resolve('/[season]/[group]/matches/[game]', { season, group, game: String(game.gameId) })}/"
								>
									<time datetime={game.date}><span class="year">{game.date.slice(0, 4)}/</span>{game.date.slice(5).replace('-', '/')}</time>
									<span class="sr">對{game.opponent}的單場頁</span>
								</a>
							</th>
							<td class="opponent">
								{game.opponent}
								<small>{game.side === 'home' ? '主場' : '客場'}</small>
								{#if game.stage && game.stage !== '例行賽'}<small>{game.stage}</small>{/if}
								{#if game.playedGoldenSet}<small>黃金決勝局</small>{/if}
							</td>
							<td class="result">
								<strong>{game.won === null ? '—' : game.won ? '勝' : '敗'}</strong>
								{#if game.score}<small>{game.score.join(' : ')}</small>{/if}
							</td>
							<td class="num points">{game.stats.total_points}</td>
							<td class="num">{game.stats.attack_points}</td>
							<td class="num">{game.stats.block_points}</td>
							<td class="num">{game.stats.serve_points}</td>
							{#each details as col (col.label)}
								<td class="num wide">{col.value(game.stats)}</td>
							{/each}
							<td class="toggle">
								<button
									type="button" aria-label="{game.date} 對{game.opponent}的詳細數據"
									aria-controls={id} aria-expanded={!!open[game.gameId]}
									onclick={() => open[game.gameId] = !open[game.gameId]}
								>
									<Icon name="chevron-down" size={16} />
								</button>
							</td>
						</tr>
						<tr class="detail" class:open={open[game.gameId]} {id}>
							<td colspan="8">
								{#if open[game.gameId]}
									<dl>
										{#each details as col (col.label)}
											<div><dt>{col.label}</dt><dd>{col.value(game.stats)}</dd></div>
										{/each}
									</dl>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<p id="{uid}-note">接發球、防守、舉球顯示到位／總數。各項數字不含黃金決勝局。</p>
		</div>
	</section>
{/if}

<style>
	section {
		container-type: inline-size;
		margin-block-start: var(--space-5);
	}

	h2 {
		margin: 0 0 var(--space-3);
	}

	.log-card {
		padding: var(--space-2);
	}

	table {
		width: 100%;
		border-collapse: collapse;
	}

	th, td {
		text-align: start;
		padding: var(--space-2) var(--space-1);
	}

	thead th, small, dt, p {
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--text-muted);
	}

	thead th {
		white-space: nowrap;
	}

	.main {
		position: relative;
		border-block-start: 1px solid var(--line);
		height: 3rem;
	}

	.main:hover, .main:focus-within {
		background: var(--surface-2);
	}

	a {
		color: var(--accent);
		font-size: var(--text-sm);
		white-space: nowrap;
	}

	a::after {
		content: '';
		position: absolute;
		inset: 0;
	}

	.main:hover a, a:focus-visible {
		text-decoration: underline;
	}

	.opponent {
		font-weight: 700;
		line-height: 1.35;
		overflow-wrap: anywhere;
	}

	small {
		display: block;
	}

	.result {
		white-space: nowrap;
	}

	.num {
		text-align: end;
	}

	td.num {
		color: var(--text-muted);
		font-weight: 600;
	}

	td.points {
		color: var(--text);
		font-weight: var(--weight-num);
		font-size: var(--num-md);
	}

	.toggle {
		width: var(--tap);
		padding: 0;
	}

	button {
		position: relative;
		display: grid;
		place-items: center;
		width: var(--tap);
		height: var(--tap);
		border: 0;
		border-radius: var(--radius-full);
		background: transparent;
		color: var(--text-muted);
		cursor: pointer;
	}

	button:hover {
		color: var(--accent);
	}

	button[aria-expanded='true'] :global(svg) {
		transform: rotate(180deg);
	}

	.wide, .year, .detail {
		display: none;
	}

	.detail.open {
		display: table-row;
	}

	.detail td {
		background: var(--surface-2);
		padding: var(--space-3);
	}

	dl {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--space-2);
		margin: 0;
	}

	dd {
		margin: 0;
		font-weight: var(--weight-num);
	}

	p {
		margin: 0;
		padding: var(--space-3) var(--space-1) var(--space-1);
		border-block-start: 1px solid var(--line);
	}

	.sr {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip-path: inset(50%);
	}

	@container (min-width: 21rem) and (max-width: 26rem) {
		.log-card {
			padding-inline: var(--space-1);
		}
		th, td {
			padding-inline: 2px;
		}
		.opponent {
			white-space: nowrap;
		}
		thead th.num {
			white-space: normal;
		}
	}

	@container (min-width: 48rem) {
		.log-card {
			padding: var(--space-3) var(--space-4);
		}
		.wide {
			display: table-cell;
		}
		.year {
			display: inline;
		}
		.toggle, .detail.open {
			display: none;
		}
	}
</style>
