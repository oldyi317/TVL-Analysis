<script module lang="ts">
	let registered = false;

	async function loadEcharts() {
		const echarts = await import('echarts/core');
		if (registered) return echarts;
		const [{ LineChart, BarChart }, { GridComponent, TooltipComponent, LegendComponent }, { CanvasRenderer }, { chartTheme }] =
			await Promise.all([
				import('echarts/charts'),
				import('echarts/components'),
				import('echarts/renderers'),
				import('./theme.js')
			]);
		echarts.use([LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);
		echarts.registerTheme('tvl-light', chartTheme('light'));
		echarts.registerTheme('tvl-dark', chartTheme('dark'));
		registered = true;
		return echarts;
	}

	type Obj = Record<string, unknown>;
	type ECBasicOption = import('echarts/core').EChartsCoreOption;

	function withDefaults(option: object): ECBasicOption {
		const o = option as Obj;
		const grid = o.grid;
		const legend: Obj = { bottom: 0, ...(o.legend as Obj) };
		const bottom = legend.show === false ? 8 : 40;
		return {
			...o,
			tooltip: { confine: true, ...(o.tooltip as Obj) },
			legend,
			grid: Array.isArray(grid) ? grid : { top: 24, right: 16, bottom, left: 8, containLabel: true, ...(grid as Obj) }
		};
	}
</script>

<script lang="ts">
	import { onMount } from 'svelte';
	import type { ECharts } from 'echarts/core';

	let {
		option,
		height = '16rem',
		ariaLabel
	}: { option: object; height?: string; ariaLabel: string } = $props();

	let el: HTMLDivElement;
	let chart: ECharts | undefined;

	$effect(() => {
		chart?.setOption(withDefaults(option), true);
	});

	onMount(() => {
		const query = matchMedia('(prefers-color-scheme: dark)');
		let destroyed = false;
		let echarts: Awaited<ReturnType<typeof loadEcharts>> | undefined;

		function init() {
			if (!echarts) return;
			chart?.dispose();
			chart = echarts.init(el, query.matches ? 'tvl-dark' : 'tvl-light');
			chart.setOption(withDefaults(option));
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
