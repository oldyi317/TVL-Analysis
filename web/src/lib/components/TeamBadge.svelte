<script lang="ts">
	import { textColorOn } from '#lib/color.js';
	import type { Team } from '#lib/data/types.js';

	let { team }: { team: Team } = $props();

	// 四字以內單行；更長的在中英交界或中點斷成兩行，避免「屏東台／電」這種斷法
	function lines(name: string): string[] {
		if (name.length <= 4) return [name];
		const boundary = name.search(/[A-Za-z0-9]/);
		const cut = boundary > 0 ? boundary : Math.ceil(name.length / 2);
		return [name.slice(0, cut), name.slice(cut)];
	}
	const parts = $derived(lines(team.short_name));
</script>

<!-- 隊標是裝飾性的，球隊名稱由旁邊的文字承載 -->
<span class="badge" style:background={team.color} style:color={textColorOn(team.color)} aria-hidden="true">
	{#each parts as part, i (i)}
		<span>{part}</span>
	{/each}
</span>

<style>
	.badge {
		display: inline-flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		inline-size: 3rem;
		block-size: 3rem;
		flex: none;
		border-radius: 50%;
		font-size: 0.625rem;
		font-weight: 600;
		line-height: 1.2;
		letter-spacing: 0.02em;
		white-space: nowrap;
	}
</style>
