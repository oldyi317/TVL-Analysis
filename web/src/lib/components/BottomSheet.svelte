<script lang="ts">
	import type { Snippet } from 'svelte';

	let { label, children }: { label: string; children: Snippet } = $props();

	let dialog = $state<HTMLDialogElement>();
	let trigger: HTMLElement | undefined;

	// trigger：關閉後把焦點還給它（Safari 點按鈕不會讓按鈕取得焦點，不能靠瀏覽器還原）
	export function open(from?: HTMLElement) {
		trigger = from;
		dialog?.showModal();
	}

	function onclick(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if (target === dialog || target.closest('a')) dialog?.close();
	}
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
<dialog bind:this={dialog} aria-label={label} {onclick} onclose={() => trigger?.focus()}>
	<div class="panel">
		{@render children()}
	</div>
</dialog>

<style>
	dialog {
		position: fixed;
		inset: 0;
		inline-size: 100%;
		block-size: 100%;
		max-inline-size: none;
		max-block-size: none;
		margin: 0;
		padding: 0;
		border: 0;
		background: var(--scrim);
		color: var(--text);
	}

	dialog[open] {
		display: flex;
		align-items: flex-end;
	}

	dialog::backdrop {
		background: transparent;
	}

	.panel {
		inline-size: 100%;
		max-block-size: 80dvh;
		overflow-y: auto;
		padding: var(--space-3) calc(var(--space-4) + var(--safe-right)) calc(var(--space-4) + var(--safe-bottom))
			calc(var(--space-4) + var(--safe-left));
		background: var(--surface);
		border-start-start-radius: var(--radius-lg);
		border-start-end-radius: var(--radius-lg);
		box-shadow: var(--shadow-pop);
	}
</style>
