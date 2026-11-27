<script module lang="ts">
	let registered = false;

	async function loadEcharts() {
		const echarts = await import('echarts/core');
		if (registered) return echarts;
		const [{ LineChart, BarChart, ScatterChart }, { GridComponent, TooltipComponent, LegendComponent, MarkLineComponent }, { CanvasRenderer }, { chartTheme }] =
			await Promise.all([
				import('echarts/charts'),
				import('echarts/components'),
				import('echarts/renderers'),
				import('./theme.js')
			]);
		echarts.use([LineChart, BarChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, CanvasRenderer]);
		echarts.registerTheme('tvl-light', chartTheme('light'));
		echarts.registerTheme('tvl-dark', chartTheme('dark'));
		registered = true;
		return echarts;
	}
</script>

<script lang="ts">
	import { onMount } from 'svelte';
	import type { ECharts } from 'echarts/core';
	import type { ChartOption } from './spec.js';

	let {
		option,
		height = '16rem',
		ariaLabel
	}: { option: ChartOption; height?: string; ariaLabel: string } = $props();

	let el: HTMLDivElement;
	let chart: ECharts | undefined;

	$effect(() => {
		// 先讀 option 建立依賴；chart 尚未初始化時若用 ?. 短路，effect 會沒有依賴而永不重跑
		const next = option.option;
		chart?.setOption(next, true);
	});

	onMount(() => {
		const query = matchMedia('(prefers-color-scheme: dark)');
		let destroyed = false;
		let echarts: Awaited<ReturnType<typeof loadEcharts>> | undefined;

		function init() {
			if (!echarts) return;
			chart?.dispose();
			chart = echarts.init(el, query.matches ? 'tvl-dark' : 'tvl-light');
			chart.setOption(option.option);
		}

		loadEcharts().then((loaded) => {
			if (destroyed) return;
			echarts = loaded;
			init();
		});
		query.addEventListener('change', init);
		const observer = new ResizeObserver(() => chart?.resize());
		observer.observe(el);

		return () => {
			destroyed = true;
			query.removeEventListener('change', init);
			observer.disconnect();
			chart?.dispose();
			chart = undefined;
		};
	});
</script>

<div bind:this={el} role="img" aria-label={ariaLabel} style:block-size={height}></div>

<style>
	div {
		inline-size: 100%;
	}
</style>
