<script lang="ts">
	import {
		distributionOption,
		isRateMetric,
		METRIC_KEYS,
		METRIC_LABELS,
		noPrReason as prReason,
		PR_METRICS,
		trendOption,
		type GameLogEntry,
		type MetricKey
	} from '#lib/advanced.js';
	import Chart from '#lib/chart/Chart.svelte';
	import DataNote from '#lib/components/DataNote.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import { positionLabel } from '#lib/site.js';
	import type { PlayerAdvancedEntry, PositionDistribution } from '#lib/data/types.js';

	let {
		advanced,
		distribution,
		gameLog
	}: {
		advanced: PlayerAdvancedEntry | null;
		distribution: PositionDistribution | null;
		gameLog: GameLogEntry[];
	} = $props();

	const uid = $props.id();
	const panelId = `${uid}-panel`;
	let open = $state(false);
	let metric = $state<MetricKey>('attack_success_rate');

	const positionText = $derived(
		advanced?.position ? positionLabel(advanced.position, true) : ''
	);
	const noPrReason = $derived(prReason(advanced));
	const fmt = (key: MetricKey, value: number) => (isRateMetric(key) ? `${value}%` : String(value));
	const distOption = $derived(
		distribution
			? distributionOption(distribution.metrics[metric], advanced?.ratios[metric] ?? null, METRIC_LABELS[metric])
			: null
	);
	const trend = $derived(trendOption(gameLog));
</script>

<h2>
	<button type="button" aria-expanded={open ? 'true' : 'false'} aria-controls={panelId} onclick={() => (open = !open)}>
		進階數據
		<span class="chevron" aria-hidden="true"></span>
	</button>
</h2>

<div id={panelId}>
	{#if open}
		<section>
			<h3>PR 值</h3>
			{#if noPrReason || !advanced?.pr}
				<DataNote text={noPrReason ?? '本季沒有進階數據'} />
			{:else}
				<p class="sub">與本季同組別 {positionText} {distribution?.player_count ?? '—'} 位球員比較</p>
				<table aria-label="PR 值">
					<thead>
						<tr>
							<th scope="col">指標</th>
							<th scope="col" class="num">數值</th>
							<th scope="col" class="num">PR 值</th>
						</tr>
					</thead>
					<tbody>
						{#each PR_METRICS as key (key)}
							<tr>
								<th scope="row">{METRIC_LABELS[key]}</th>
								<td class="num">{fmt(key, advanced.ratios[key])}</td>
								<td class="num strong">{advanced.pr[key] ?? '—'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		</section>

		{#if distOption}
			<section>
				<h3>同位置分佈</h3>
				<div class="metrics" role="group" aria-label="選擇指標">
					{#each METRIC_KEYS as key (key)}
						<button type="button" aria-pressed={metric === key ? 'true' : 'false'} onclick={() => (metric = key)}>
							{METRIC_LABELS[key]}
						</button>
					{/each}
				</div>
				<Chart option={distOption} ariaLabel="{METRIC_LABELS[metric]}在同位置球員中的分佈長條圖" />
				<p class="sub">每根長條是一位球員，由低到高排列；深色是這位球員，虛線是中位數。</p>
			</section>
		{/if}

		<section>
			<h3>逐場趨勢</h3>
			{#if gameLog.length === 0}
				<EmptyState title="本季尚無逐場數據" />
			{:else}
				<Chart option={trend} ariaLabel="逐場得分與攻擊成功率趨勢折線圖" />
				{#if gameLog.some((e) => e.golden)}
					<p class="sub">菱形標記的場次另有黃金決勝局，其數據未計入。</p>
				{/if}
			{/if}
		</section>
	{/if}
</div>

<style>
	h2 {
		font-size: var(--text-lg);
		font-weight: 700;
		margin-block: var(--space-6) var(--space-3);
	}

	h2 button {
		display: flex;
		align-items: center;
		justify-content: space-between;
		inline-size: 100%;
		min-block-size: var(--tap);
		padding: 0;
		background: none;
		border: 0;
		color: inherit;
		font: inherit;
		text-align: start;
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

	h2 button[aria-expanded='true'] .chevron {
		transform: translateY(25%) rotate(-135deg);
	}

	section {
		margin-block-end: var(--space-4);
		padding: var(--space-3);
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: var(--radius-md);
	}

	h3 {
		margin: 0 0 var(--space-2);
		font-size: var(--text-md);
		font-weight: 700;
	}

	.sub {
		margin: 0 0 var(--space-2);
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	table {
		inline-size: 100%;
		border-collapse: collapse;
		font-size: var(--text-md);
	}

	th,
	td {
		padding: var(--space-2) var(--space-1);
		border-block-start: 1px solid var(--line);
		font-weight: 400;
		text-align: start;
	}

	thead th {
		border-block-start: 0;
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.num {
		text-align: end;
	}

	td.strong {
		font-weight: var(--weight-num);
	}

	.metrics {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-2);
		margin-block-end: var(--space-2);
	}

	.metrics button {
		min-block-size: var(--tap);
		padding: 0 var(--space-3);
		background: none;
		border: 1px solid var(--line);
		border-radius: var(--radius-md);
		color: var(--text-muted);
		font: inherit;
		font-size: var(--text-sm);
		cursor: pointer;
	}

	.metrics button[aria-pressed='true'] {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--on-accent);
	}
</style>
