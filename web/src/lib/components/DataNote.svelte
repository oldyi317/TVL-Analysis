<script lang="ts">
	let { text }: { text: string } = $props();
</script>

<p role="note">{text}</p>

<style>
	p {
		margin: 0;
		padding: var(--space-2) var(--space-3);
		background: var(--warn-tint);
		color: var(--warn-ink);
		border-radius: var(--radius-sm);
	}
</style>
