<script lang="ts">
	import { onMount } from 'svelte';
	import { applyUpdate, generatedMD, isStandalone, proto, realStandalone, refreshInfo, resetAll, setCfg, startProto } from './proto.svelte.js';

	let open = $state(false);

	onMount(() => {
		void startProto();
		const timer = setInterval(() => open && void refreshInfo(), 1000);
		return () => clearInterval(timer);
	});

	const fmt = (t: number) => new Date(t).toLocaleTimeString('zh-TW', { hour12: false });
	const offline = $derived(!proto.online || proto.cfg.offline);
	const choices = {
		navTimeout: [1000, 2000, 3000, 5000],
		delay: [0, 2000, 5000, 10000],
		threshold: [30 * 60 * 1000, 60 * 1000, 10 * 1000]
	};
	const label = (ms: number) => (ms >= 60000 ? `${ms / 60000} 分鐘` : `${ms / 1000} 秒`);
</script>

{#if proto.pending}
	<button type="button" class="bar update" data-testid="proto-update" onclick={applyUpdate}>有新資料，點此更新</button>
{:else if offline}
	<div class="bar off" data-testid="proto-offbar">離線中，資料更新至 {generatedMD}</div>
{/if}

<button type="button" class="fab" data-testid="proto-fab" aria-label="除錯面板" onclick={() => (open = !open)}>除錯</button>

{#if open}
	<div class="panel" data-testid="proto-panel" role="dialog" aria-label="除錯面板">
		<div class="head"><b>離線原型除錯面板</b><button type="button" data-testid="proto-close" onclick={() => (open = false)}>關閉</button></div>
		<dl>
			<dt>頁面版本</dt><dd data-testid="page-version">{proto.pageVersion}</dd>
			<dt>接手的 SW 版本</dt><dd data-testid="sw-version">{proto.swVersion || '（無）'}</dd>
			<dt>等待中的新版</dt><dd>{proto.hasWaiting ? '有' : '無'}</dd>
			<dt>最近查到的最新版</dt><dd>{proto.latest || '（未查）'}</dd>
			<dt>換版狀態</dt><dd>{proto.status || '—'}</dd>
			<dt>顯示模式</dt><dd>{isStandalone() ? '獨立視窗' : '瀏覽器分頁'}{proto.cfg.standalone && !realStandalone() ? '（除錯開關）' : ''}</dd>
			<dt>網路</dt><dd>{proto.online ? '線上' : '離線'}{proto.cfg.offline ? '（模擬離線）' : ''}</dd>
			<dt>這次導覽</dt><dd data-testid="nav-how">{proto.nav ? `${proto.nav.how}，${proto.nav.ms} ms` : '—'}</dd>
			<dt>程式檔快取</dt><dd>{proto.preCount} 項</dd>
			<dt>資料檔快取</dt><dd>{proto.dataCount} 項</dd>
			<dt>當季預先下載</dt>
			<dd data-testid="predl">
				{#if proto.predl}{proto.predl.done}／{proto.predl.total}{proto.predl.failed ? `（失敗 ${proto.predl.failed}）` : ''}{proto.predl.running ? ' 進行中' : ''}{proto.predl.version !== proto.swVersion ? '（舊版）' : ''}{:else}未開始{/if}
			</dd>
		</dl>

		<h4>設定</h4>
		<div class="row"><span>導覽等待上限</span>
			{#each choices.navTimeout as v (v)}<button type="button" class:on={proto.cfg.navTimeout === v} onclick={() => setCfg({ navTimeout: v })}>{v / 1000} 秒</button>{/each}
		</div>
		<div class="row"><span>人為延遲導覽回應</span>
			{#each choices.delay as v (v)}<button type="button" class:on={proto.cfg.delay === v} onclick={() => setCfg({ delay: v })}>{v / 1000} 秒</button>{/each}
		</div>
		<div class="row"><span>模擬離線</span>
			<button type="button" class:on={proto.cfg.offline} onclick={() => setCfg({ offline: true })}>開</button>
			<button type="button" class:on={!proto.cfg.offline} onclick={() => setCfg({ offline: false })}>關</button>
		</div>
		<div class="row"><span>自動換新門檻</span>
			{#each choices.threshold as v (v)}<button type="button" class:on={proto.cfg.threshold === v} onclick={() => setCfg({ threshold: v })}>{label(v)}</button>{/each}
		</div>
		<div class="row"><span>當成獨立視窗</span>
			<button type="button" class:on={proto.cfg.standalone} onclick={() => setCfg({ standalone: true })}>開</button>
			<button type="button" class:on={!proto.cfg.standalone} onclick={() => setCfg({ standalone: false })}>關</button>
		</div>
		<div class="row"><button type="button" class="danger" onclick={resetAll}>清掉離線資料並註銷</button></div>

		<h4>事件紀錄（新的在上）</h4>
		<ol class="log" data-testid="log">
			{#each proto.log as e (e.t + e.ev)}
				<li>{fmt(e.t)} {e.ev}{e.away !== undefined ? ` 離開 ${Math.round(e.away / 1000)} 秒` : ''}{e.nav ? `｜${e.nav}` : ''}{e.note ? `｜${e.note}` : ''}</li>
			{/each}
		</ol>
	</div>
{/if}

<style>
	.bar {
		position: fixed;
		inset-block-start: 0;
		inset-inline: 0;
		z-index: 9000;
		padding: 0.35rem 1rem;
		padding-block-start: calc(0.35rem + env(safe-area-inset-top, 0px));
		text-align: center;
		font-size: 0.85rem;
		border: 0;
		font-family: inherit;
	}
	.update {
		background: #1c1c1a;
		color: #fff;
		cursor: pointer;
	}
	.off {
		background: #b45309;
		color: #fff;
	}
	.fab {
		position: fixed;
		inset-inline-end: 0.5rem;
		inset-block-end: calc(var(--tabbar-h, 3.5rem) + env(safe-area-inset-bottom, 0px) + 0.5rem);
		z-index: 9000;
		padding: 0.35rem 0.6rem;
		border-radius: 999px;
		border: 1px solid #888;
		background: rgb(255 255 255 / 0.9);
		color: #111;
		font-size: 0.75rem;
	}
	.panel {
		position: fixed;
		inset-inline: 0.5rem;
		inset-block: 2.5rem calc(var(--tabbar-h, 3.5rem) + env(safe-area-inset-bottom, 0px) + 2.8rem);
		z-index: 9001;
		overflow: auto;
		padding: 0.75rem;
		background: #fff;
		color: #111;
		border: 1px solid #888;
		border-radius: 8px;
		font-size: 0.8rem;
		line-height: 1.4;
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	dl {
		display: grid;
		grid-template-columns: max-content 1fr;
		gap: 0.15rem 0.75rem;
		margin: 0.5rem 0;
	}
	dd {
		margin: 0;
		word-break: break-all;
	}
	h4 {
		margin: 0.6rem 0 0.25rem;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.3rem;
		margin-block: 0.3rem;
	}
	.row span {
		min-inline-size: 8.5rem;
	}
	.row button,
	.head button {
		padding: 0.3rem 0.6rem;
		border: 1px solid #888;
		border-radius: 6px;
		background: #f3f3f3;
		color: #111;
	}
	.row button.on {
		background: #1c1c1a;
		color: #fff;
	}
	.row button.danger {
		background: #b91c1c;
		color: #fff;
	}
	.log {
		margin: 0;
		padding-inline-start: 1.2rem;
	}
</style>
