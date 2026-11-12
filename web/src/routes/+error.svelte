<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';

	// 原型：service worker 對「沒存過、又抓不到」的 __data.json 回 503 offline-miss
	const offlineMiss = $derived(page.status === 503 && page.error?.message === 'offline-miss');

	function back() {
		if (history.length > 1) history.back();
		else location.href = resolve('/');
	}
</script>

{#if offlineMiss}
	<div class="miss" data-testid="offline-miss">
		<h1>這一頁離線時看不到，連上網路後再試</h1>
		<button type="button" onclick={back}>返回</button>
	</div>
{:else}
	<h1>{page.status}</h1>
	<p>{page.error?.message}</p>
{/if}

<style>
	.miss h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin-block: 1.5rem 0.75rem;
	}

	.miss button {
		min-block-size: var(--tap);
		padding-inline: 1.25rem;
		font: inherit;
		font-weight: 500;
	}
</style>
