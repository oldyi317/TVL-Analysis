<script lang="ts">
	let { value, size = 56, label }: { value: number; size?: number; label?: string } = $props();

	const stroke = $derived(Math.max(5, Math.round(size / 8.8)));
	const radius = $derived((size - stroke) / 2);
	const circumference = $derived(2 * Math.PI * radius);
	const offset = $derived(circumference * (1 - Math.min(Math.max(value, 0), 100) / 100));
</script>

<span class="ring" style="inline-size:{size}px;block-size:{size}px;--c:{circumference};--o:{offset}" aria-hidden="true">
	<svg width={size} height={size} viewBox="0 0 {size} {size}">
		<circle class="track" cx={size / 2} cy={size / 2} r={radius} stroke-width={stroke} />
		<circle
			class="bar"
			cx={size / 2}
			cy={size / 2}
			r={radius}
			stroke-width={stroke}
			stroke-dasharray={circumference}
			stroke-dashoffset={offset}
			transform="rotate(-90 {size / 2} {size / 2})"
		/>
	</svg>
	<span class="center">
		<b class:big={size >= 80}>{value}%</b>
		{#if label && size >= 80}<small>{label}</small>{/if}
	</span>
</span>

<style>
	.ring {
		position: relative;
		display: inline-grid;
		flex: none;
	}

	svg {
		grid-area: 1 / 1;
	}

	.track,
	.bar {
		fill: none;
	}

	.track {
		stroke: var(--accent-tint);
	}

	.bar {
		stroke: var(--accent);
		stroke-linecap: round;
		animation: fill 600ms ease-out both;
	}

	@keyframes fill {
		from {
			stroke-dashoffset: var(--c);
		}
		to {
			stroke-dashoffset: var(--o);
		}
	}

	.center {
		grid-area: 1 / 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		line-height: 1.1;
	}

	b {
		font-size: var(--text-sm);
		font-weight: var(--weight-num);
		color: var(--text);
	}

	b.big {
		font-size: var(--num-md);
	}

	small {
		font-size: 0.75rem;
		color: var(--text-muted);
	}
</style>
