<script module lang="ts">
	import type { BoxRow } from '#lib/boxScore.js';
	import { positionLabel } from '#lib/site.js';

	const goodOverTotal = (good: number, total: number) => `${good}／${total}`;

	// label 用在展開列，short 用在表頭；沒有 short 就共用 label
	const DETAIL_COLS: { label: string; short?: string; value: (r: BoxRow) => string }[] = [
		{ label: '攻擊次數', value: (r) => String(r.attack_total) },
		{ label: '發球次數', value: (r) => String(r.serve_total) },
		{ label: '接發', value: (r) => goodOverTotal(r.receive_excellent, r.receive_total) },
		{ label: '防守', value: (r) => goodOverTotal(r.dig_excellent, r.dig_total) },
		{ label: '舉球', value: (r) => goodOverTotal(r.set_excellent, r.set_total) },
		{ label: '出賽局數', short: '局數', value: (r) => String(r.sets_played ?? '—') }
	];
</script>

<script lang="ts">
	import DataNote from '#lib/components/DataNote.svelte';
	let { rows, label }: { rows: BoxRow[]; label: string } = $props();

	const uid = $props.id();
	let open = $state<Record<number, boolean>>({});

	const detailId = (row: BoxRow) => `${uid}-${row.player_id}`;
	const toggle = (row: BoxRow) => (open[row.player_id] = !open[row.player_id]);
</script>

{#if rows.some((row) => row.sets_played === null)}
	<DataNote text="官網單場數據不提供出賽局數" />
{/if}

<div class="scroll">
	<table aria-label={label}>
		<thead>
			<tr>
				<th scope="col" class="num">背號</th>
				<th scope="col" class="player">球員</th>
				<th scope="col" class="num">得分</th>
				<th scope="col" class="num">攻擊</th>
				<th scope="col" class="num">攔網</th>
				<th scope="col" class="num">發球</th>
				{#each DETAIL_COLS as col (col.label)}
					<th scope="col" class="num wide">{col.short ?? col.label}</th>
				{/each}
				<th scope="col" class="toggle"><span class="sr">詳細</span></th>
			</tr>
		</thead>
		<tbody>
			{#each rows as row (row.player_id)}
				<tr class="main">
					<td class="num">{row.number ?? '—'}</td>
					<td class="player">
						<div class="who">
							<span class="name">{row.name}</span>
							<span class="pos">{positionLabel(row.position)}</span>
						</div>
					</td>
					<td class="num strong">{row.total_points}</td>
					<td class="num">{row.attack_points}</td>
					<td class="num">{row.block_points}</td>
					<td class="num">{row.serve_points}</td>
					{#each DETAIL_COLS as col (col.label)}
						<td class="num wide">{col.value(row)}</td>
					{/each}
					<td class="toggle">
						<button
							type="button"
							aria-expanded={open[row.player_id] ? 'true' : 'false'}
							aria-controls={detailId(row)}
							aria-label="{row.name}詳細數據"
							onclick={() => toggle(row)}
						>
							<span class="chevron" aria-hidden="true"></span>
						</button>
					</td>
				</tr>
				<tr class="detail" class:open={open[row.player_id]} id={detailId(row)}>
					<td colspan="7">
						<dl>
							{#each DETAIL_COLS as col (col.label)}
								<div>
									<dt>{col.label}</dt>
									<dd><b>{col.value(row)}</b></dd>
								</div>
							{/each}
						</dl>
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.scroll {
		overflow-x: auto;
	}

	table {
		inline-size: 100%;
		border-collapse: collapse;
		font-size: var(--text-md);
	}

	th {
		padding: var(--space-2) var(--space-1);
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--text-muted);
		text-align: start;
		white-space: nowrap;
		border-block-end: 1px solid var(--line);
	}

	td {
		padding: var(--space-2) var(--space-1);
	}

	.main td {
		block-size: 3rem;
		border-block-start: 1px solid var(--line);
	}

	tbody tr.main:first-child td {
		border-block-start: 0;
	}

	.num {
		text-align: end;
	}

	td.num {
		font-size: var(--text-md);
		font-weight: 600;
		line-height: 1;
		color: var(--text-muted);
		white-space: nowrap;
	}

	td.strong {
		font-size: var(--num-md);
		font-weight: var(--weight-num);
		color: var(--text);
	}

	td.player {
		max-inline-size: 0;
		inline-size: 100%;
	}

	.who {
		display: flex;
		align-items: baseline;
		gap: var(--space-1);
	}

	.name {
		flex: 1 1 auto;
		font-weight: 700;
		min-inline-size: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.pos {
		flex: none;
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.toggle {
		inline-size: var(--tap);
		padding: 0;
	}

	button {
		display: flex;
		border-radius: var(--radius-full);
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
		transition: transform 0.15s;
	}

	button[aria-expanded='true'] .chevron {
		transform: translateY(25%) rotate(-135deg);
	}

	.detail {
		display: none;
	}

	.detail.open {
		display: table-row;
	}

	.detail td {
		background: var(--surface-2);
		padding: var(--space-2) var(--space-4) var(--space-4);
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

	dt {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	dd {
		margin: 0;
	}

	dd b {
		font-size: var(--text-lg);
		font-weight: var(--weight-num);
		line-height: 1.2;
	}

	.sr {
		position: absolute;
		inline-size: 1px;
		block-size: 1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
	}

	.wide {
		display: none;
	}

	@media (min-width: 48rem) {
		.wide {
			display: table-cell;
		}

		.toggle,
		.detail,
		.detail.open {
			display: none;
		}
	}
</style>
