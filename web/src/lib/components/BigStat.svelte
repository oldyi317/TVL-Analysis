<script lang="ts">
	let {
		label,
		value,
		detail = null,
		tone = 1
	}: {
		label: string;
		value: number | string | null;
		detail?: string | null;
		tone?: 1 | 2 | 3;
	} = $props();
</script>

<div class="stat-tile tone-{tone}">
	<p class="label">{label}</p>
	<p class="value">{value ?? '—'}</p>
	{#if detail}
		<p class="detail">{detail}</p>
	{/if}
</div>

<style>
	.tone-1 {
		--tile: var(--tile-1);
	}

	.tone-2 {
		--tile: var(--tile-2);
	}

	.tone-3 {
		--tile: var(--tile-3);
	}

	p {
		margin: 0;
	}

	.label {
		font-size: var(--text-sm);
		font-weight: 700;
		color: rgba(255, 255, 255, 0.9);
	}

	.detail {
		font-size: var(--text-sm);
		color: rgba(255, 255, 255, 0.9);
	}

	.value {
		font-size: var(--num-lg);
		font-weight: var(--weight-num);
		line-height: 1.1;
	}
</style>
