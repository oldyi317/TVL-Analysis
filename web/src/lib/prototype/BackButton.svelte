<script lang="ts">
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import Icon from '#lib/components/Icon.svelte';
	import { shell } from './shell.svelte.js';

	// 連結算法照各深層頁既有的 listHref
	const listHref = $derived.by(() => {
		const { season, group, game, team, player } = page.params;
		if (!season || !group) return undefined;
		const p = { season, group };
		if (game) {
			const date = (page.data as { match?: { date: string } }).match?.date;
			return `${resolve('/[season]/[group]/matches', p)}/${date ? `#m-${date.slice(0, 7)}` : ''}`;
		}
		if (team) return `${resolve('/[season]/[group]/teams', p)}/`;
		if (player) return `${resolve('/[season]/[group]/players', p)}/`;
		return undefined;
	});

	function back() {
		if (shell.depth > 0) history.back();
		else if (listHref) goto(listHref);
	}
</script>

<button type="button" aria-label="返回" onclick={back}><Icon name="chevron-left" size={24} /></button>

<style>
	button {
		flex: none;
		display: grid;
		place-items: center;
		inline-size: 44px;
		block-size: 44px;
		padding: 0;
		border: 0;
		border-radius: var(--radius-full);
		background: transparent;
		color: var(--text);
		cursor: pointer;
	}

	button:active {
		background: var(--accent-tint);
	}
</style>
