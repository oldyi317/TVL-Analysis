<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { OFFLINE_MISS } from '#lib/offline.js';

	const offlineMiss = $derived(page.status === 503 && page.error?.message === OFFLINE_MISS);

	// 直接離線開到沒存的頁時沒有上一頁，退回首頁
	function back() {
		if (history.length > 1) history.back();
		else void goto(resolve('/'));
	}
</script>

{#if offlineMiss}
	<SiteHead title="離線時看不到這一頁｜VStat 企排" />
	<h1 class="miss">這一頁離線時看不到，連上網路後再試</h1>
	<button type="button" class="chip back" onclick={back}>返回</button>
{:else}
	<h1>{page.status}</h1>
	<p>{page.error?.message}</p>
{/if}

<style>
	.miss {
		font-size: var(--text-xl);
		font-weight: 700;
		margin-block: 1.5rem 0.75rem;
	}

	.back {
		background: var(--accent-tint);
		color: var(--accent);
		font-weight: 700;
	}
</style>
