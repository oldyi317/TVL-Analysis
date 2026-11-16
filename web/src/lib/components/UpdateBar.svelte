<script lang="ts">
	import { applyUpdate, update } from '#lib/serviceWorker.svelte.js';
</script>

<!-- live region 要先在 DOM 裡，內容出現時輔助技術才會報讀 -->
<div class="bar" role="status">
	{#if update.ready}
		<button type="button" class="chip" onclick={() => applyUpdate()}>有新資料，點此更新</button>
	{/if}
</div>

<style>
	.bar {
		position: fixed;
		inset-block-end: calc(var(--tabbar-h) + var(--safe-bottom) + var(--space-1));
		inset-inline: 0;
		z-index: var(--z-tabbar);
		display: flex;
		justify-content: center;
		padding-inline: calc(var(--space-4) + var(--safe-left)) calc(var(--space-4) + var(--safe-right));
		pointer-events: none;
	}

	button {
		pointer-events: auto;
		background: var(--accent);
		color: var(--on-accent);
		box-shadow: var(--shadow-pop);
	}

	button:hover {
		background: var(--accent-strong);
	}

	@media (min-width: 64rem) {
		.bar {
			inset-block-end: var(--space-5);
			inset-inline-start: var(--sidebar);
		}
	}
</style>
