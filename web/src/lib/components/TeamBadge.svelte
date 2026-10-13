<script lang="ts">
	import type { Team } from '#lib/data/types.js';
	import { teamStyle } from './teamStyle.js';

	let { team, size = 'md' }: { team: Team; size?: 'sm' | 'md' | 'lg' } = $props();

	// 三字以內單行；四字拆 2+2；更長的在中英交界或中點斷成兩行
	function lines(name: string): string[] {
		if (name.length <= 3) return [name];
		if (name.length === 4) return [name.slice(0, 2), name.slice(2)];
		const boundary = name.search(/[A-Za-z0-9]/);
		const cut = boundary > 0 ? boundary : Math.ceil(name.length / 2);
		return [name.slice(0, cut), name.slice(cut)];
	}
	const parts = $derived(lines(team.short_name));
</script>

<!-- 隊標是裝飾性的，球隊名稱由旁邊的文字承載 -->
<span class="badge {size}" style={teamStyle(team)} aria-hidden="true">
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
		background: var(--team-fill);
		color: var(--team-on-fill);
		font-size: 0.75rem;
		font-weight: 600;
		line-height: 1.1;
		letter-spacing: 0.02em;
		white-space: nowrap;
	}

	.sm {
		inline-size: 2rem;
		block-size: 2rem;
		font-size: 0.625rem;
		letter-spacing: 0;
	}

	.lg {
		inline-size: 4rem;
		block-size: 4rem;
		font-size: 1rem;
	}
</style>
