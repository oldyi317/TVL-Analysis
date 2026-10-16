<script lang="ts">
	let {
		label,
		value,
		detail = null
	}: { label: string; value: number | string | null; detail?: string | null } = $props();
</script>

<div>
	<p class="label">{label}</p>
	<p class="value">{value ?? '—'}</p>
	{#if detail}
		<p class="detail">{detail}</p>
	{/if}
</div>

<style>
	p {
		margin: 0;
	}

	.label {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.detail {
		margin-block-start: var(--space-1);
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.value {
		font-size: var(--num-xl);
		font-weight: var(--weight-num);
		line-height: 1;
	}
</style>
