<script lang="ts">
	import {
		postgameMetricRows,
		postgameSourceRows,
		type PostgameComparison,
		type PostgameRow
	} from '#lib/teamAdvanced.js';

	let {
		comparison,
		homeName,
		awayName
	}: {
		comparison: PostgameComparison;
		homeName: string;
		awayName: string;
	} = $props();

	const uid = $props.id();
	const sourceRows = $derived(postgameSourceRows(comparison));
	const metricRows = $derived(postgameMetricRows(comparison));
</script>

{#snippet comparisonTable(id: string, title: string, firstCol: string, rows: PostgameRow[])}
	<section aria-labelledby={id}>
		<h3 {id}>{title}</h3>
		<div class="card table-card">
			<table aria-label={title}>
				<thead>
					<tr>
						<th scope="col">{firstCol}</th>
						<th scope="col" class="num">{homeName}</th>
						<th scope="col" class="num">{awayName}</th>
					</tr>
				</thead>
				<tbody>
					{#each rows as row (row.key)}
						<tr class:total={row.key === 'total'} class:tied={row.leader === null}>
							<th scope="row">{row.label}</th>
							<td class="num" class:strong={row.leader === 'home'}>
								{#if row.leader === 'home'}<strong>{row.home}</strong>{:else}{row.home}{/if}
							</td>
							<td class="num" class:strong={row.leader === 'away'}>
								{#if row.leader === 'away'}<strong>{row.away}</strong>{:else}{row.away}{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</section>
{/snippet}

<div class="groups">
	{@render comparisonTable(`${uid}-sources`, '得分來源', '項目', sourceRows)}
	{@render comparisonTable(`${uid}-metrics`, '球隊指標', '指標', metricRows)}
</div>

<style>
	.groups {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--space-5);
		align-items: start;
	}

	@media (min-width: 64rem) {
		.groups {
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: var(--space-4);
		}
	}

	h3 {
		margin-block: 0 var(--space-3);
	}

	.table-card {
		padding-block: var(--space-2);
		overflow-x: auto;
	}

	table {
		inline-size: 100%;
		border-collapse: collapse;
	}

	th,
	td {
		block-size: 3rem;
		padding: 0 var(--space-2);
		border-block-start: 1px solid var(--line);
	}

	thead th {
		border-block-start: 0;
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--text-muted);
		text-align: start;
		white-space: nowrap;
	}

	th.num,
	td.num {
		text-align: end;
		white-space: nowrap;
	}

	th[scope='row'] {
		text-align: start;
		font-size: var(--text-md);
		font-weight: 600;
		color: var(--text);
	}

	td.num {
		font-size: var(--text-md);
		font-weight: 600;
		color: var(--text-muted);
	}

	.total th[scope='row'],
	.total.tied td.num {
		font-weight: 700;
		color: var(--text);
	}

	td.strong {
		font-size: var(--text-lg);
		font-weight: var(--weight-num);
		color: var(--text);
	}

	td.strong strong {
		font-weight: inherit;
	}
</style>
