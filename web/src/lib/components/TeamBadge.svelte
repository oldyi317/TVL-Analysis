<script lang="ts">
	import { asset } from '$app/paths';
	import type { AssetPath } from '$app/types';
	import type { SeasonTeam } from '#lib/site.js';
	import { teamStyle } from './teamStyle.js';

	let {
		team,
		size = 'md',
		label
	}: { team: SeasonTeam; size?: 'sm' | 'md' | 'lg' | 'xl'; label?: string } = $props();

	// 三字以內單行；四字拆 2+2；更長的在中英交界或中點斷成兩行
	function lines(name: string): string[] {
		if (name.length <= 3) return [name];
		if (name.length === 4) return [name.slice(0, 2), name.slice(2)];
		const boundary = name.search(/[A-Za-z0-9]/);
		const cut = boundary > 0 ? boundary : Math.ceil(name.length / 2);
		return [name.slice(0, cut), name.slice(cut)];
	}
	const parts = $derived(lines(team.short_name));

	// 隊徽路徑由 seasonTeam 決定；沒有圖或載入失敗就退回隊色圓形
	const logoSrc = $derived(team.logo === null ? null : asset(team.logo as AssetPath));
	let failedSrc = $state('');
	const failed = $derived(logoSrc === null || failedSrc === logoSrc);
	// 有給 label 才由圓標自己承載名稱；落回縮寫文字時改由外層說明
	const labelled = $derived(label !== undefined);
	const describesItself = $derived(labelled && failed);
</script>

<!-- 沒給 label：隊色圓標是裝飾性的，球隊名稱由旁邊的文字承載，整顆 aria-hidden。
     有給 label：圓標自己承載名稱，有圖時由 img 的 alt 說明，落回縮寫文字時由外層 role="img" 說明 -->
<span
	class="badge {size}"
	class:has-logo={!failed}
	style={teamStyle(team)}
	aria-hidden={labelled ? undefined : 'true'}
	role={describesItself ? 'img' : undefined}
	aria-label={describesItself ? label : undefined}
>
	{#if failed}
		{#each parts as part, i (i)}
			<span>{part}</span>
		{/each}
	{:else}
		<img src={logoSrc} alt={label ?? ''} onerror={() => (failedSrc = logoSrc ?? '')} />
	{/if}
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
		overflow: hidden;
	}

	.has-logo {
		background: #ffffff;
		box-shadow: inset 0 0 0 1px var(--line);
	}

	img {
		inline-size: 82%;
		block-size: 82%;
		object-fit: contain;
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

	.xl {
		inline-size: 6rem;
		block-size: 6rem;
		font-size: 1.5rem;
	}

</style>
