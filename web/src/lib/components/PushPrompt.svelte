<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import Icon from '#lib/components/Icon.svelte';
	import { browserStorage } from '#lib/groupMemory.js';
	import { push, refreshPush } from '#lib/push.svelte.js';
	import { readPromptHidden, writePromptHidden } from '#lib/pushPrefs.js';
	import { isStandalone } from '#lib/serviceWorker.svelte.js';

	// 獨立視窗與關閉記號在瀏覽器裡才知道；權限與推播訂閱每次掛載都重新問瀏覽器，問完才決定顯不顯示
	let asked = $state(false);
	const show = $derived(asked && push.support === true && push.permission === 'default' && !push.subscribed);

	function close() {
		writePromptHidden(browserStorage());
		asked = false;
	}

	onMount(() => {
		if (!isStandalone() || readPromptHidden(browserStorage())) return;
		void refreshPush().then(() => (asked = true));
	});
</script>

{#if show}
	<div class="card push-prompt rise">
		<p>想在有新賽果、比賽日早上收到通知嗎？</p>
		<a class="chip" href="{resolve('/notifications')}/">設定通知</a>
		<button type="button" aria-label="關閉這則提示" onclick={close}><Icon name="close" /></button>
	</div>
{/if}

<style>
	.push-prompt {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		grid-template-areas:
			'text close'
			'link close';
		align-items: start;
		gap: var(--space-1) var(--space-2);
		margin-block-end: var(--space-5);
		padding-block: var(--space-3);
		padding-inline-end: var(--space-2);
	}

	p {
		grid-area: text;
		display: flex;
		align-items: center;
		min-block-size: var(--tap);
		margin: 0;
	}

	a {
		grid-area: link;
		justify-self: start;
		background: var(--accent-tint);
		color: var(--accent);
		font-size: var(--text-sm);
		font-weight: 700;
	}

	button {
		grid-area: close;
		display: grid;
		place-items: center;
		inline-size: var(--tap);
		block-size: var(--tap);
		padding: 0;
		border: 0;
		border-radius: var(--radius-full);
		background: transparent;
		color: var(--text-muted);
		cursor: pointer;
		transition:
			background-color 160ms ease-out,
			color 160ms ease-out;
	}

	button:hover {
		background: var(--surface-2);
		color: var(--text);
	}

	@media (min-width: 40rem) {
		.push-prompt {
			grid-template-columns: minmax(0, 1fr) auto auto;
			grid-template-areas: 'text link close';
			align-items: center;
		}
	}
</style>
