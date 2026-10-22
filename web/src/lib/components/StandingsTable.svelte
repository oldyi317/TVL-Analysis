<script lang="ts">
	import type { SetRecordKey, StandingsRow } from '#lib/site.js';
	import DataNote from './DataNote.svelte';
	import TeamBadge from './TeamBadge.svelte';

	const RECORD_KEYS: SetRecordKey[] = ['3-0', '3-1', '3-2', '2-3', '1-3', '0-3'];

	function ratio(value: number | null): string {
		return value === null ? '—' : value.toFixed(3);
	}

	let { rows, note }: { rows: StandingsRow[]; note?: string } = $props();

	const uid = $props.id();
	let open = $state<Record<string, boolean>>({});

	const keyOf = (row: StandingsRow) => `${row.team.group}-${row.team.team_id}`;
	const detailId = (row: StandingsRow) => `${uid}-${keyOf(row)}`;
	const toggle = (row: StandingsRow) => (open[keyOf(row)] = !open[keyOf(row)]);
</script>

{#snippet last5Tags(row: StandingsRow)}
	{#if row.last5.length === 0}
		—
	{:else}
		<span class="tags">
			{#each row.last5 as result, i (i)}
				<span class="tag" class:w={result === 'W'} class:l={result === 'L'}>
					{result === 'W' ? '勝' : '敗'}
				</span>
			{/each}
		</span>
	{/if}
{/snippet}

{#if note}
	<div class="note"><DataNote text={note} /></div>
{/if}

<div class="standings">
<div class="card">
<div class="scroll">
	<table>
		<thead>
			<tr>
				<th scope="col" class="rank">名次</th>
				<th scope="col" class="team">隊伍</th>
				<th scope="col" class="num">勝</th>
				<th scope="col" class="num">敗</th>
				<th scope="col" class="num wide">勝率</th>
				<th scope="col" class="num">積分</th>
				{#each RECORD_KEYS as key (key)}
					<th scope="col" class="num wide">{key}</th>
				{/each}
				<th scope="col" class="num wide">勝局</th>
				<th scope="col" class="num wide">敗局</th>
				<th scope="col" class="num wide">局數比</th>
				<th scope="col" class="num wide">得分</th>
				<th scope="col" class="num wide">失分</th>
				<th scope="col" class="num wide">得失分比</th>
				<th scope="col" class="wide">近五場</th>
				<th scope="col" class="toggle"><span class="sr">詳細</span></th>
			</tr>
		</thead>
		<tbody>
			{#each rows as row (keyOf(row))}
				<tr class="main" class:expanded={open[keyOf(row)]}>
					<td class="rank"><span class="pos" class:top={row.rank <= 3}>{row.rank}</span></td>
					<td class="team">
						<span class="team-cell">
							<TeamBadge team={row.team} size="sm" />
							<span class="name">{row.team.short_name}</span>
						</span>
					</td>
					<td class="num wins">{row.wins}</td>
					<td class="num">{row.losses}</td>
					<td class="num wide small">{ratio(row.win_pct)}</td>
					<td class="num">{row.points}</td>
					{#each RECORD_KEYS as key (key)}
						<td class="num wide small">{row.set_record[key]}</td>
					{/each}
					<td class="num wide small">{row.sets_won}</td>
					<td class="num wide small">{row.sets_lost}</td>
					<td class="num wide small">{ratio(row.set_ratio)}</td>
					<td class="num wide small">{row.points_for}</td>
					<td class="num wide small">{row.points_against}</td>
					<td class="num wide small">{ratio(row.point_ratio)}</td>
					<td class="wide">{@render last5Tags(row)}</td>
					<td class="toggle">
						<button
							type="button"
							aria-expanded={open[keyOf(row)] ? 'true' : 'false'}
							aria-controls={detailId(row)}
							aria-label="{row.team.short_name}詳細數據"
							onclick={() => toggle(row)}
						>
							<span class="chevron" aria-hidden="true"></span>
						</button>
					</td>
				</tr>
				<tr class="detail" class:open={open[keyOf(row)]} id={detailId(row)}>
					<td colspan="6">
						<dl>
							<div class="full">
								<dt>局數比分布</dt>
								<dd class="records">
									{#each RECORD_KEYS as key (key)}
										<span><span class="k">{key}</span> <b>{row.set_record[key]}</b></span>
									{/each}
								</dd>
							</div>
							<div>
								<dt>勝率</dt>
								<dd><b>{ratio(row.win_pct)}</b></dd>
							</div>
							<div>
								<dt>局數比</dt>
								<dd><b>{ratio(row.set_ratio)}</b></dd>
							</div>
							<div>
								<dt>勝局／敗局</dt>
								<dd><b>{row.sets_won}／{row.sets_lost}</b></dd>
							</div>
							<div>
								<dt>得失分比</dt>
								<dd><b>{ratio(row.point_ratio)}</b></dd>
							</div>
							<div>
								<dt>得分／失分</dt>
								<dd><b>{row.points_for}／{row.points_against}</b></dd>
							</div>
							<div class="full">
								<dt>近五場</dt>
								<dd>{@render last5Tags(row)}</dd>
							</div>
						</dl>
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
</div>
</div>

<style>
	.note {
		margin-block-end: var(--space-3);
	}

	.card {
		min-inline-size: 0;
		max-inline-size: 100%;
		padding: var(--space-2) var(--space-3);
	}

	.scroll {
		min-inline-size: 0;
		max-inline-size: 100%;
		overflow-x: auto;
	}

	table {
		inline-size: 100%;
		border-collapse: collapse;
		font-size: var(--text-md);
	}

	th {
		padding: var(--space-2);
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--text-muted);
		text-align: start;
		white-space: nowrap;
		border-block-end: 1px solid var(--line);
	}

	td {
		block-size: 3rem;
		padding: var(--space-1) var(--space-2);
	}

	.main td {
		border-block-end: 1px solid var(--line);
	}

	tbody tr.main:nth-last-child(2) td,
	tbody tr.main.expanded td {
		border-block-end: 0;
	}

	.rank {
		text-align: center;
	}

	.num {
		text-align: end;
	}

	td.num {
		font-weight: 600;
	}

	td.wins {
		font-weight: 800;
	}

	td.small {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.pos {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		inline-size: 1.5rem;
		block-size: 1.5rem;
		border-radius: 50%;
		font-weight: 700;
	}

	.pos.top {
		background: var(--accent-tint);
		color: var(--accent);
	}

	.team-cell {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		white-space: nowrap;
	}

	.name {
		font-weight: 700;
	}

	.toggle {
		inline-size: var(--tap);
		padding: 0;
	}

	button {
		display: flex;
		align-items: center;
		justify-content: center;
		inline-size: var(--tap);
		block-size: var(--tap);
		padding: 0;
		background: none;
		border: 0;
		color: var(--text-muted);
		cursor: pointer;
	}

	.chevron {
		inline-size: 0.5rem;
		block-size: 0.5rem;
		border-inline-end: 2px solid currentColor;
		border-block-end: 2px solid currentColor;
		transform: translateY(-25%) rotate(45deg);
		transition: transform 160ms ease-out;
	}

	button[aria-expanded='true'] .chevron {
		transform: translateY(25%) rotate(-135deg);
	}

	.detail {
		display: none;
		background: var(--surface-2);
	}

	.detail.open {
		display: table-row;
	}

	.detail td {
		block-size: auto;
		padding: var(--space-3) var(--space-4) var(--space-4);
		border-end-start-radius: var(--radius-md);
		border-end-end-radius: var(--radius-md);
	}

	dl {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--space-3);
		margin: 0;
	}

	dl > div {
		min-inline-size: 0;
	}

	.full {
		grid-column: 1 / -1;
	}

	dt {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	dd {
		margin: 0;
	}

	dd b {
		font-size: var(--num-md);
		font-weight: var(--weight-num);
		line-height: 1.2;
	}

	.records {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: var(--space-1) var(--space-3);
	}

	.k {
		color: var(--text-muted);
	}

	.tags {
		display: inline-flex;
		gap: var(--space-1);
	}

	.tags .tag {
		padding: 0 var(--space-2);
	}

	.tag.w {
		background: var(--win-tint);
		color: var(--win);
	}

	.tag.l {
		background: var(--loss-tint);
		color: var(--loss);
	}

	.sr {
		position: absolute;
		inline-size: 1px;
		block-size: 1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
	}

	.standings {
		container: standings / inline-size;
	}

	.wide {
		display: none;
	}

	@container standings (min-width: 44rem) {
		.card {
			padding: var(--space-3) var(--space-4);
		}

		.wide {
			display: table-cell;
			padding-inline: 0.375rem;
		}

		.toggle,
		.detail,
		.detail.open {
			display: none;
		}

		tbody tr.main:nth-last-child(2) td {
			border-block-end: 0;
		}
	}
</style>
