<script lang="ts">
	let { text }: { text: string } = $props();
</script>

<p role="note">{text}</p>

<style>
	p {
		margin: 0;
		padding: var(--space-3) var(--space-4);
		background: var(--warn-tint);
		color: var(--warn-ink);
		font-size: var(--text-md);
		border-radius: var(--radius-md);
	}
</style>
