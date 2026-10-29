<script lang="ts">
	import { resolve } from '$app/paths';
	import SiteHead from '#lib/components/SiteHead.svelte';
</script>

<SiteHead title="找不到這頁｜TVL 數據" />

<h1>找不到這頁</h1>
<a class="home" href={resolve('/')}>回首頁</a>

<style>
	h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin-block: 1.5rem 0.75rem;
	}

	.home {
		display: inline-flex;
		align-items: center;
		min-block-size: var(--tap);
		font-weight: 500;
	}
</style>
