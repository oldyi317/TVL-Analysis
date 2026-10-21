<script lang="ts">
	import {
		defaultTrendMetric,
		distributionOption,
		isRateMetric,
		METRIC_KEYS,
		METRIC_LABELS,
		noPrReason as prReason,
		PR_METRICS,
		TREND_METRICS,
		trendLabel,
		trendOption,
		type GameLogEntry,
		type MetricKey
	} from '#lib/advanced.js';
	import { onMount } from 'svelte';
	import Chart from '#lib/chart/Chart.svelte';
	import DataNote from '#lib/components/DataNote.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import Icon from '#lib/components/Icon.svelte';
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
	onMount(() => {
		if (window.matchMedia('(min-width: 64rem)').matches) open = true;
	});
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
	let trendMetric = $derived(defaultTrendMetric(advanced?.position ?? null));
	const trend = $derived(trendOption(gameLog, trendMetric));
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

		{#if distOption}
			<section class="card">
				<h3>同位置分佈</h3>
				<div class="metrics" role="group" aria-label="選擇分佈指標">
					{#each METRIC_KEYS as key (key)}
						<button type="button" class="chip" aria-pressed={metric === key ? 'true' : 'false'} onclick={() => (metric = key)}>
							{METRIC_LABELS[key]}
						</button>
					{/each}
				</div>
				<Chart option={distOption} ariaLabel="{METRIC_LABELS[metric]}在同位置球員中的分佈長條圖" />
				<p class="sub">每根長條是一位球員，由低到高排列；深色是這位球員，虛線是中位數。</p>
			</section>
		{/if}

		<section class="card">
			<h3>逐場趨勢</h3>
			{#if gameLog.length === 0}
				<EmptyState title="本季尚無單場數據" />
			{:else}
				<div class="metrics" role="group" aria-label="選擇逐場趨勢指標">
					{#each TREND_METRICS as key (key)}
						<button type="button" class="chip" aria-pressed={trendMetric === key ? 'true' : 'false'} onclick={() => (trendMetric = key)}>
							{trendLabel(key)}
						</button>
					{/each}
				</div>
				<Chart option={trend} ariaLabel="{trendLabel(trendMetric)}逐場趨勢折線圖" />
				{#if gameLog.some((e) => e.golden)}
					<p class="sub">菱形標記的場次另有黃金決勝局，其數據未計入。</p>
				{/if}
			{/if}
		</section>
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

	.metrics button {
		border: 0;
		font: inherit;
		font-size: var(--text-sm);
		cursor: pointer;
	}

	.metrics button[aria-pressed='true'] {
		background: var(--accent);
		color: var(--on-accent);
	}
</style>
