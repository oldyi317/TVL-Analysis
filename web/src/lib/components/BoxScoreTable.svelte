<script module lang="ts">
	import { detailCols, type BoxRow } from '#lib/boxScore.js';
	import { positionLabel } from '#lib/site.js';

	const DETAIL_COLS = detailCols({ receiveLabel: '接發' });
</script>

<script lang="ts">
	let { rows, label }: { rows: BoxRow[]; label: string } = $props();

	const uid = $props.id();
	let open = $state<Record<number, boolean>>({});

	const detailId = (row: BoxRow) => `${uid}-${row.player_id}`;
	const toggle = (row: BoxRow) => (open[row.player_id] = !open[row.player_id]);
</script>

<div class="scroll">
	<table class="expand-table" aria-label={label}>
		<thead>
			<tr>
				<th scope="col" class="num">背號</th>
				<th scope="col" class="player">球員</th>
				<th scope="col" class="num">得分</th>
				<th scope="col" class="num">攻擊</th>
				<th scope="col" class="num">攔網</th>
				<th scope="col" class="num">發球</th>
				{#each DETAIL_COLS as col (col.label)}
					<th scope="col" class="num wide">{col.label}</th>
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
					<td class="num points">{row.total_points}</td>
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
		font-size: var(--text-md);
	}

	th {
		border-block-end: 1px solid var(--line);
	}

	.main td {
		block-size: 3rem;
		border-block-start: 1px solid var(--line);
	}

	tbody tr.main:first-child td {
		border-block-start: 0;
	}

	td.num {
		line-height: 1;
		white-space: nowrap;
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

	.detail td {
		padding: var(--space-2) var(--space-4) var(--space-4);
	}

	dl {
		gap: var(--space-3);
	}

	dl > div {
		min-inline-size: 0;
	}

	dd b {
		font-size: var(--text-lg);
		font-weight: var(--weight-num);
		line-height: 1.2;
	}

	@media (min-width: 48rem) {
		table {
			--wide-cols: table-cell;
			--expander: none;
		}
	}
</style>
