<script lang="ts">
	type SegmentItem = { key: string; label: string; href: string | undefined; current: boolean };

	// stay：同一頁內切換檢視（如紀錄類型），換頁後保留捲動位置與焦點
	let { label, items, stay = false }: { label: string; items: SegmentItem[]; stay?: boolean } = $props();
</script>

<nav aria-label={label}>
	{#each items as item (item.key)}
		<a
			href={item.href}
			aria-current={item.current ? 'page' : undefined}
			data-sveltekit-noscroll={stay || undefined}
			data-sveltekit-keepfocus={stay || undefined}
		>
			{item.label}
		</a>
	{/each}
</nav>

<style>
	nav {
		display: flex;
		inline-size: 100%;
		gap: var(--space-1);
		padding: var(--space-1);
		background: var(--surface-2);
		border-radius: var(--radius-full);
	}

	a {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		min-block-size: calc(var(--tap) - var(--space-2));
		font-size: var(--text-md);
		font-weight: 700;
		color: var(--text-muted);
		border-radius: var(--radius-full);
		transition: color 160ms ease-out;
	}

	a:hover {
		color: var(--text);
	}

	a[aria-current='page'] {
		background: var(--surface);
		color: var(--accent);
		box-shadow: var(--shadow-card);
		outline: 1px solid var(--card-edge);
		outline-offset: -1px;
	}

	a:focus-visible {
		outline-color: var(--accent);
		outline-offset: 2px;
	}
</style>
