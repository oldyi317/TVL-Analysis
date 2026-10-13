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
		border: 1px solid var(--line);
		border-radius: var(--radius-sm);
		overflow: hidden;
	}

	a {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		min-block-size: var(--tap);
		font-size: var(--text-md);
		font-weight: 700;
		color: var(--text-muted);
	}

	a[aria-current='page'] {
		background: var(--accent);
		color: var(--on-accent);
		border-radius: var(--radius-sm);
	}
</style>
