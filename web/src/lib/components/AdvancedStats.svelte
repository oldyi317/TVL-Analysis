<script lang="ts">
	import {
		availableMetrics,
		defaultTrendMetric,
		distributionOption,
		distributionSummary,
		formatMetricValue,
		METRIC_KEYS,
		METRIC_LABELS,
		noPrReason as prReason,
		PR_METRICS,
		TREND_METRICS,
		trendLabel,
		trendLineOption,
		trendPoints,
		trendSummary,
		type GameLogEntry,
		type TrendMetric,
		type MetricKey
	} from '#lib/advanced.js';
	import { onMount } from 'svelte';
	import ChartFrame from '#lib/components/ChartFrame.svelte';
	import DataNote from '#lib/components/DataNote.svelte';
	import Icon from '#lib/components/Icon.svelte';
	import { positionLabel } from '#lib/site.js';
	import type { PlayerAdvancedEntry, PositionDistribution } from '#lib/data/types.js';

	let {
		advanced,
		distribution,
		gameLog,
		position,
		setsPlayed
	}: {
		advanced: PlayerAdvancedEntry | null;
		distribution: PositionDistribution | null;
		gameLog: GameLogEntry[];
		position: string | null;
		setsPlayed: number | null;
	} = $props();

	const uid = $props.id();
	const panelId = `${uid}-panel`;
	let open = $state(false);
	onMount(() => {
		if (window.matchMedia('(min-width: 64rem)').matches) open = true;
	});
	const metricKeys = $derived(availableMetrics(METRIC_KEYS, setsPlayed, position));
	const prKeys = $derived(availableMetrics(PR_METRICS, setsPlayed, position));
	const trendKeys = $derived(availableMetrics(TREND_METRICS, setsPlayed, position));
	let selectedMetric = $state<MetricKey>('attack_success_rate');
	const metric = $derived(metricKeys.includes(selectedMetric) ? selectedMetric : 'attack_success_rate');

	const positionText = $derived(
		advanced?.position ? positionLabel(advanced.position, true) : ''
	);
	const noPrReason = $derived(prReason(advanced));
	const fmt = formatMetricValue;
	const distOption = $derived(
		distribution
			? distributionOption(distribution.metrics[metric], advanced?.ratios[metric] ?? null, METRIC_LABELS[metric])
			: null
	);
	let selectedTrend = $state<TrendMetric | null>(null);
	const trendMetric = $derived(
		selectedTrend && trendKeys.includes(selectedTrend) ? selectedTrend : defaultTrendMetric(position, setsPlayed)
	);
	const points = $derived(trendPoints(gameLog, trendMetric));
	const trend = $derived(points.length ? trendLineOption(points, trendMetric) : null);
	const distSummary = $derived(distribution
		? distributionSummary(distribution.metrics[metric], advanced?.ratios[metric] ?? null, metric)
		: null);
</script>

<h2 class="card">
	<button type="button" aria-expanded={open ? 'true' : 'false'} aria-controls={panelId} onclick={() => (open = !open)}>
		進階數據
		<span class="chevron"><Icon name="chevron-left" /></span>
	</button>
</h2>

<div id={panelId}>
	{#if open}
		<section class="card">
			<h3>PR 值</h3>
			{#if noPrReason}
				<DataNote text={noPrReason} />
			{:else if advanced}
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
						{#each prKeys as key (key)}
							<tr>
								<th scope="row">{METRIC_LABELS[key]}</th>
								<td class="num">{fmt(key, advanced.ratios[key])}</td>
								<td class="num strong">
									<span class="pr">
										{#if advanced.pr[key] !== null && advanced.pr[key] !== undefined}
											<span class="track" aria-hidden="true"><span class="fill" style:inline-size="{advanced.pr[key]}%"></span></span>
										{/if}
										<span class="pr-value">{advanced.pr[key] ?? '—'}</span>
									</span>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		</section>

		{#if distOption && distribution}
			<ChartFrame
				title="同位置分佈"
				games={gameLog.length}
				conclusion={distSummary}
				label={distSummary ?? undefined}
				option={distOption}
				emptyTitle="本季沒有同位置分佈數據"
				note="每根長條是一位球員，由低到高排列；深色是這位球員或同值球員，虛線是中位數。"
			>
				{#snippet controls()}
					<div class="metrics" role="group" aria-label="選擇分佈指標">
						{#each metricKeys as key (key)}
							<button type="button" class="chip" aria-pressed={metric === key ? 'true' : 'false'} onclick={() => (selectedMetric = key)}>
								{METRIC_LABELS[key]}
							</button>
						{/each}
					</div>
				{/snippet}
				{#snippet table()}
					<table style:font-size="var(--text-md)" aria-label="同位置分佈：{METRIC_LABELS[metric]}">
						<caption>{distSummary}</caption>
						<thead><tr><th scope="col">由低到高位置</th><th scope="col" class="num">{METRIC_LABELS[metric]}</th><th scope="col">標記</th></tr></thead>
						<tbody>
							{#each distribution.metrics[metric].values as value, i (i)}
								<tr>
									<th scope="row">{i + 1}</th>
									<td class="num">{fmt(metric, value)}</td>
									<td>{value === advanced?.ratios[metric] ? '本人或同值球員' : '—'}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				{/snippet}
			</ChartFrame>
		{/if}

		<ChartFrame
			title="逐場趨勢"
			games={gameLog.length}
			conclusion={null}
			label={trendSummary(points, trendMetric)}
			option={trend}
			emptyTitle="本季尚無單場數據"
			note={gameLog.some((e) => e.golden) ? '菱形標記的場次另有黃金決勝局，其數據未計入。' : undefined}
		>
			{#snippet controls()}
				{#if gameLog.length}
					<div class="metrics" role="group" aria-label="選擇逐場趨勢指標">
						{#each trendKeys as key (key)}
							<button type="button" class="chip" aria-pressed={trendMetric === key ? 'true' : 'false'} onclick={() => (selectedTrend = key)}>
								{trendLabel(key)}
							</button>
						{/each}
					</div>
				{/if}
			{/snippet}
			{#snippet table()}
				<table style:font-size="var(--text-md)" aria-label="逐場趨勢：{trendLabel(trendMetric)}">
					<thead><tr><th scope="col">日期</th><th scope="col" class="num">{trendLabel(trendMetric)}</th><th scope="col">說明</th></tr></thead>
					<tbody>
						{#each points as point, i (i)}
							<tr>
								<th scope="row">{point.date}</th>
								<td class="num">{fmt(trendMetric, point.value)}</td>
								<td>{point.value === null ? '無數據' : point.detail ?? '—'}{point.golden ? '；此場另有黃金決勝局，未計入' : ''}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/snippet}
		</ChartFrame>
	{/if}
</div>

<style>
	h2 {
		margin-block: var(--space-5) var(--space-3);
		padding-block: 0;
		font-size: var(--text-lg);
		font-weight: 700;
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
		display: flex;
		color: var(--text-muted);
		transform: rotate(-90deg);
		transition: transform 0.15s;
	}

	h2 button[aria-expanded='true'] .chevron {
		transform: rotate(90deg);
	}

	section {
		margin-block-end: var(--space-4);
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

	tbody th {
		font-weight: 600;
	}

	thead th {
		border-block-start: 0;
		border-block-end: 1px solid var(--line);
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--text-muted);
	}

	tbody tr:first-child th,
	tbody tr:first-child td {
		border-block-start: 0;
	}

	.num {
		text-align: end;
	}

	td.strong {
		inline-size: 45%;
		font-weight: var(--weight-num);
	}

	.pr {
		display: flex;
		align-items: center;
		justify-content: flex-end;
		gap: var(--space-2);
	}

	.track {
		flex: 1;
		max-inline-size: 7rem;
		block-size: 6px;
		border-radius: var(--radius-full);
		background: var(--accent-tint);
		overflow: hidden;
	}

	.fill {
		display: block;
		block-size: 100%;
		border-radius: var(--radius-full);
		background: var(--accent);
	}

	.pr-value {
		min-inline-size: 2ch;
		text-align: end;
	}

	.metrics {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-1);
		margin-block-end: var(--space-2);
	}
</style>
