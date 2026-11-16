<script lang="ts">
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { installPrompt, promptInstall } from '#lib/installPrompt.svelte.js';
	import { NOTIFICATIONS_TITLE } from '#lib/nav.js';

	const guide = $derived(installPrompt.guide);
</script>

<SiteHead title="{NOTIFICATIONS_TITLE}｜VStat 企排" />

<h1>{NOTIFICATIONS_TITLE}</h1>

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
