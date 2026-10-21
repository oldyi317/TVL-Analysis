<script lang="ts">
	import { GROUPS, GROUP_LABEL, type Group } from '#lib/site.js';

	let {
		current,
		hrefFor
	}: { current: Group; hrefFor: (group: Group) => string | undefined } = $props();
</script>

<nav aria-label="組別">
	{#each GROUPS as group (group)}
		<a href={hrefFor(group)} aria-current={current === group ? 'page' : undefined}>
			{GROUP_LABEL[group]}
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
