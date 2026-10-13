<script lang="ts">
	let { label, value }: { label: string; value: number | string | null } = $props();
</script>

<div>
	<p class="label">{label}</p>
	<p class="value">{value ?? '—'}</p>
</div>

<style>
	p {
		margin: 0;
	}

	.label {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.value {
		font-size: var(--num-xl);
		font-weight: var(--weight-num);
		line-height: 1;
	}
</style>
