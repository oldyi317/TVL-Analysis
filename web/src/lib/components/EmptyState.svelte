<script lang="ts">
	let { title, hint }: { title: string; hint?: string } = $props();
</script>

<div>
	<p class="title">{title}</p>
	{#if hint}
		<p class="hint">{hint}</p>
	{/if}
</div>

<style>
	div {
		padding: var(--space-6);
		text-align: center;
		color: var(--text-muted);
	}

	p {
		margin: 0;
	}

	.title {
		font-size: var(--text-md);
	}

	.hint {
		font-size: var(--text-md);
	}
</style>
