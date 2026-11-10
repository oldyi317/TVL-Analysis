<script lang="ts">
	// 原型專用，不做 production gate：它要用 production build 的 preview 給手機看，且分支不會合併。
	// 要回答的問題（#188）：頂部橫捲 pill 導覽在 iPhone 獨立視窗下夠不夠用？深層頁能否靠站內元件回上一層？要不要改底部分頁？
	import { onMount } from 'svelte';
	import { LABELS, ORDER, setVariant, shell, syncVariant } from './shell.svelte.js';

	let ready = $state(false);
	let open = $state(false);
	let probe = $state<HTMLElement>();
	let readout = $state({ standalone: '', nav: '', safe: '', win: '' });

	function measure() {
		const mm = matchMedia('(display-mode: standalone)').matches;
		const nav = (navigator as { standalone?: boolean }).standalone;
		const cs = probe ? getComputedStyle(probe) : undefined;
		readout = {
			standalone: `mq=${mm} nav=${nav}`,
			nav: String(shell.depth),
			safe: cs ? `上${cs.paddingTop} 右${cs.paddingRight} 下${cs.paddingBottom} 左${cs.paddingLeft}` : '',
			win: `${innerWidth}×${innerHeight}`
		};
	}

	onMount(() => {
		syncVariant();
		ready = true;
	});

	$effect(() => {
		if (open) {
			shell.depth;
			measure();
		}
	});

	function step(d: number) {
		const i = ORDER.indexOf(shell.variant);
		setVariant(ORDER[(i + d + ORDER.length) % ORDER.length]);
	}

	function onKey(e: KeyboardEvent) {
		const t = e.target as HTMLElement | null;
		if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
		if (e.key === 'ArrowLeft') step(-1);
		else if (e.key === 'ArrowRight') step(1);
	}
</script>

<svelte:window onkeydown={onKey} onresize={() => open && measure()} />

<div class="probe" bind:this={probe} aria-hidden="true"></div>

{#if ready}
	<div class="sw" data-proto-switcher>
		{#if open}
			<dl>
				<dt>獨立視窗</dt><dd>{readout.standalone}</dd>
				<dt>安全區</dt><dd>{readout.safe}</dd>
				<dt>視窗</dt><dd>{readout.win}</dd>
				<dt>歷史深度</dt><dd>{readout.nav}</dd>
			</dl>
		{/if}
		<div class="pill">
			<button type="button" aria-label="上一個變體" onclick={() => step(-1)}>‹</button>
			<button type="button" class="label" aria-expanded={open} onclick={() => (open = !open)}>
				{LABELS[shell.variant]}
			</button>
			<button type="button" aria-label="下一個變體" onclick={() => step(1)}>›</button>
		</div>
	</div>
{/if}

<style>
	.probe {
		position: fixed;
		visibility: hidden;
		pointer-events: none;
		inline-size: 0;
		block-size: 0;
		padding: var(--safe-top) var(--safe-right) var(--safe-bottom) var(--safe-left);
	}

	.sw {
		position: fixed;
		inset-inline: 0;
		inset-block-end: calc(var(--tabbar-h) + var(--safe-bottom) + 8px);
		z-index: 10;
		display: grid;
		justify-items: center;
		gap: 6px;
		pointer-events: none;
	}

	.sw > * {
		pointer-events: auto;
	}

	.pill {
		display: flex;
		align-items: center;
		border-radius: 999px;
		background: #111;
		color: #fff;
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
		font-family: var(--font);
	}

	.pill button {
		min-inline-size: 44px;
		block-size: 40px;
		padding: 0 6px;
		border: 0;
		background: transparent;
		color: inherit;
		font: inherit;
		font-size: 20px;
		font-weight: 700;
	}

	.pill .label {
		padding: 0 6px;
		font-size: 14px;
	}

	dl {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 2px 10px;
		margin: 0;
		padding: 8px 12px;
		border-radius: 12px;
		background: #111;
		color: #fff;
		font: 12px/1.4 ui-monospace, monospace;
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
	}

	dt {
		color: #aaa;
	}

	dd {
		margin: 0;
	}
</style>
