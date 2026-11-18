<script lang="ts">
	import { onMount } from 'svelte';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { browserStorage } from '#lib/groupMemory.js';
	import { inPhoneBrowserTab } from '#lib/installGuide.js';
	import { installPrompt, promptInstall } from '#lib/installPrompt.svelte.js';
	import { NOTIFICATIONS_TITLE } from '#lib/nav.js';
	import {
		DENIED_TEXT,
		OFFLINE_TEXT,
		groupChecks,
		isOffline,
		isOn,
		push,
		refreshPush,
		setGroup,
		setKind
	} from '#lib/push.svelte.js';
	import { writePromptHidden } from '#lib/pushPrefs.js';

	const guide = $derived(installPrompt.guide);
	const denied = $derived(push.permission === 'denied');
	const offline = $derived(isOffline());
	const locked = $derived(denied || offline || push.busy);
	const showGroups = $derived(push.permission === 'granted' && (isOn('results') || isOn('today')));
	const checks = $derived(groupChecks());
	const status = $derived(denied ? DENIED_TEXT : offline ? OFFLINE_TEXT : push.message);

	const kinds = [
		{ kind: 'results', name: '賽果摘要', hint: '有新賽果進站時通知' },
		{ kind: 'today', name: '今日賽事', hint: '比賽日上午 10 點通知當天各場的開賽時間與賽前勝率' }
	] as const;

	// 失敗時 state 沒變，Svelte 不會重設 DOM，要自己把開關彈回原狀
	async function toggle(kind: 'results' | 'today', event: Event & { currentTarget: HTMLInputElement }) {
		const input = event.currentTarget;
		await setKind(kind, input.checked);
		input.checked = isOn(kind);
	}

	async function changeGroup(group: 'men' | 'women', event: Event & { currentTarget: HTMLInputElement }) {
		const input = event.currentTarget;
		const next = { ...checks, [group]: input.checked };
		await setGroup(next.men, next.women);
		input.checked = groupChecks()[group];
	}

	onMount(() => {
		// 進過這一頁，首頁的提示卡就不再出現
		writePromptHidden(browserStorage());
		void refreshPush();
		// 到系統設定改了通知權限再切回來，頁面不會重新載入
		const onVisible = () => {
			if (document.visibilityState === 'visible') void refreshPush();
		};
		document.addEventListener('visibilitychange', onVisible);
		return () => document.removeEventListener('visibilitychange', onVisible);
	});
</script>

<SiteHead title="{NOTIFICATIONS_TITLE}｜VStat 企排" />

<h1>{NOTIFICATIONS_TITLE}</h1>

{#snippet install()}
	{#if guide === 'ios-safari'}
		<p>在 Safari 把 VStat 企排加到主畫面：</p>
		<ol>
			<li>點分享鈕</li>
			<li>選「加入主畫面」</li>
			<li>點「加入」</li>
		</ol>
	{:else if guide === 'ios-other'}
		<p>請用 Safari 開啟本站，再加到主畫面。</p>
	{:else if guide === 'android-other'}
		<p>請用 Chrome 開啟本站，再加到主畫面。</p>
	{:else if guide === 'android-chrome'}
		{#if installPrompt.promptReady}
			<button type="button" class="chip install" onclick={promptInstall}>安裝 VStat 企排</button>
		{:else}
			<p>請從 Chrome 的選單把本站加到主畫面。已經加過的話，直接從主畫面開啟。</p>
		{/if}
	{:else if guide === 'standalone'}
		<p>已經從主畫面開啟 VStat 企排，不用再加一次。</p>
	{:else if guide === 'desktop'}
		<p>用手機瀏覽器開啟本站，就能把 VStat 企排加到主畫面。</p>
	{/if}
{/snippet}

{#if push.support === false}
	<p>加到主畫面後才能開啟通知</p>
	{@render install()}
{:else if push.support}
	<div class="switches">
		{#each kinds as { kind, name, hint } (kind)}
			<div class="row">
				<label for="push-{kind}">{name}</label>
				<p id="push-{kind}-hint">{hint}</p>
				<input
					id="push-{kind}"
					type="checkbox"
					role="switch"
					aria-describedby="push-{kind}-hint"
					checked={isOn(kind)}
					disabled={locked}
					onchange={(event) => toggle(kind, event)}
				/>
			</div>
		{/each}
	</div>

	<p class="privacy">
		開啟後，本站會存下這台裝置的推播位址（由 Apple、Google 等推播服務產生）與你選的通知種類、組別，存放在 Cloudflare。不存姓名、帳號或任何能認出你的資料。兩種通知都關掉就會刪除；推播服務回報位址失效時也會刪除。
	</p>

	{#if showGroups}
		<fieldset class="groups">
			<legend>組別</legend>
			<label>
				<input
					type="checkbox"
					checked={checks.men}
					disabled={locked || (checks.men && !checks.women)}
					onchange={(event) => changeGroup('men', event)}
				/>
				男子組
			</label>
			<label>
				<input
					type="checkbox"
					checked={checks.women}
					disabled={locked || (checks.women && !checks.men)}
					onchange={(event) => changeGroup('women', event)}
				/>
				女子組
			</label>
		</fieldset>
	{/if}

	<p class="status" role="status">{status}</p>

	{#if guide && inPhoneBrowserTab(guide)}
		<section>
			<h2>加到主畫面</h2>
			{@render install()}
		</section>
	{/if}
{/if}

<style>
	h1 {
		margin-block: var(--space-2) var(--space-4);
	}

	p,
	ol {
		margin-block: 0 var(--space-3);
	}

	ol {
		padding-inline-start: var(--space-6);
	}

	li {
		padding-block: var(--space-1);
	}

	.install {
		background: var(--accent);
		color: var(--on-accent);
	}

	.install:hover {
		background: var(--accent-strong);
	}

	.switches {
		margin-block-end: var(--space-3);
	}

	.row {
		display: grid;
		grid-template-columns: 1fr auto;
		grid-template-areas:
			'name control'
			'hint control';
		column-gap: var(--space-4);
		align-items: center;
		min-block-size: var(--tap);
		padding-block: var(--space-3);
		border-block-end: 1px solid var(--line);
	}

	.row label {
		grid-area: name;
		font-weight: 700;
	}

	.row p {
		grid-area: hint;
		margin: 0;
		color: var(--text-muted);
		font-size: var(--text-sm);
	}

	.row input {
		grid-area: control;
	}

	input[role='switch'] {
		appearance: none;
		position: relative;
		inline-size: 3rem;
		block-size: 1.75rem;
		margin: 0;
		border-radius: var(--radius-full);
		background: var(--line);
		cursor: pointer;
		transition: background 160ms ease-out;
	}

	input[role='switch']::before {
		content: '';
		position: absolute;
		inset-block-start: 0.1875rem;
		inset-inline-start: 0.1875rem;
		inline-size: 1.375rem;
		block-size: 1.375rem;
		border-radius: var(--radius-full);
		background: var(--surface);
		transition: transform 160ms ease-out;
	}

	input[role='switch']:checked {
		background: var(--accent);
	}

	input[role='switch']:checked::before {
		transform: translateX(1.25rem);
	}

	input:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.groups {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-2) var(--space-5);
		margin: 0 0 var(--space-3);
		padding: 0;
		border: 0;
	}

	.groups legend {
		padding: 0;
		margin-block-end: var(--space-2);
		font-weight: 700;
	}

	.groups label {
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
		min-block-size: var(--tap);
	}

	.status {
		min-block-size: 1.5em;
	}

	.privacy {
		color: var(--text-muted);
		font-size: var(--text-sm);
	}

	section {
		margin-block-start: var(--space-5);
	}

	h2 {
		margin-block: 0 var(--space-3);
	}

	/* 窄螢幕的頂部列已經顯示頁名：h1 只留給輔助科技 */
	@media (max-width: 63.999rem) {
		h1 {
			position: absolute;
			inline-size: 1px;
			block-size: 1px;
			overflow: hidden;
			clip-path: inset(50%);
			white-space: nowrap;
		}
	}
</style>
