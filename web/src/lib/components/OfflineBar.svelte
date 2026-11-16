<script lang="ts">
	import siteJson from '#lib/data/site.json';
	import { formatTaipeiMonthDay } from '#lib/datetime.js';
	import { connection } from '#lib/serviceWorker.svelte.js';

	const updatedAt = formatTaipeiMonthDay(siteJson.generated_at);
</script>

{#if connection.offline}
	<p role="status">離線中，資料更新至 {updatedAt}</p>
{/if}

<style>
	p {
		margin: 0 0 var(--space-4);
		padding: var(--space-2) var(--space-3);
		border-radius: var(--radius-sm);
		background: var(--warn-tint);
		color: var(--warn-ink);
		font-size: var(--text-sm);
		font-weight: 700;
	}
</style>
