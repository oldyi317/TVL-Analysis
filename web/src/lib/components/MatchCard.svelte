<script lang="ts">
	import { setsWon, winnerSide, winProbPercent, type StyleMatch } from '#lib/match.js';
	import TeamBadge from './TeamBadge.svelte';
	import WinRing from './WinRing.svelte';

	let {
		match,
		stage,
		showSets = true,
		size = 'md'
	}: { match: StyleMatch; stage?: string | null; showSets?: boolean; size?: 'md' | 'lg' } = $props();

	const played = $derived(match.sets.length > 0);
	const winner = $derived(winnerSide(match.sets));
	const [homeSets, awaySets] = $derived(setsWon(match.sets));
	const prob = $derived(winProbPercent(match.home_win_prob));
	const sides = $derived([
		{ side: 'home' as const, team: match.home, won: homeSets },
		{ side: 'away' as const, team: match.away, won: awaySets }
	]);
</script>

<article class="card {size}" aria-label="{match.home.short_name} 對 {match.away.short_name}">
	<p class="meta">
		<span>{match.date} 星期{match.weekday}{played && match.start_time ? ` ${match.start_time}` : ''} · {match.venue}</span>
		{#if stage}
			<span class="tag tag-stage">{stage}</span>
		{/if}
	</p>

	<div class="face">
		{#each sides as row, i (row.side)}
			<div class="team" class:lose={winner !== null && winner !== row.side} style="grid-column:{i === 0 ? 1 : 3};grid-row:1">
				<TeamBadge team={row.team} size={size === 'lg' ? 'lg' : 'md'} />
				<span class="who">
					<span class="name">{row.team.short_name}</span>
					{#if winner === row.side}
						<span class="tag tag-win">勝</span>
					{/if}
				</span>
			</div>
		{/each}

		<div class="mid">
			{#if played}
				<p class="result">
					<span class="score" class:dim={winner !== null && winner !== 'home'}>{homeSets}</span>
					<span class="colon">:</span>
					<span class="score" class:dim={winner !== null && winner !== 'away'}>{awaySets}</span>
				</p>
			{:else}
				<span class="tag tag-pending">未開打</span>
				{#if match.start_time}
					<span class="time">{match.start_time}</span>
				{/if}
			{/if}
		</div>
	</div>

	{#if !played}
		<div class="prob" role="group" aria-label="賽前勝率">
			{#if prob}
				<WinRing value={prob[0]} size={size === 'lg' ? 88 : 40} label="主隊" />
				<span class="prob-text">主隊 {prob[0]}% – {prob[1]}% 客隊</span>
			{:else}
				資料不足
			{/if}
		</div>
	{/if}

	{#if played && showSets}
		<p class="sets">
			{#each match.sets as [h, a], i (i)}
				<span class="set">
					<span class:strong={h > a}>{h}</span>-<span class:strong={a > h}>{a}</span>
				</span>
			{/each}
			{#if match.golden_set}
				<span class="golden">
					<span class="tag">黃金決勝局</span>
					<span><span class:strong={match.golden_set[0] > match.golden_set[1]}>{match.golden_set[0]}</span>-<span class:strong={match.golden_set[1] > match.golden_set[0]}>{match.golden_set[1]}</span></span>
				</span>
			{/if}
		</p>
	{/if}
</article>

<style>
	.meta {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-1) var(--space-3);
		margin: 0 0 var(--space-4);
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.face {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
		align-items: start;
		gap: var(--space-2);
	}

	.team {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--space-2);
		min-inline-size: 0;
		text-align: center;
		color: var(--text);
	}

	.team.lose {
		color: var(--text-muted);
	}

	.team :global(.badge.md) {
		inline-size: 3.5rem;
		block-size: 3.5rem;
	}

	.who {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: center;
		gap: var(--space-1) var(--space-2);
	}

	.name {
		font-size: var(--text-md);
		font-weight: 700;
		line-height: 1.3;
		overflow-wrap: anywhere;
	}

	.lg .name {
		font-size: var(--text-lg);
	}

	.mid {
		grid-column: 2;
		grid-row: 1;
		align-self: center;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--space-1);
		min-block-size: 3.5rem;
		justify-content: center;
	}

	.result {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin: 0;
		line-height: 1;
	}

	.score {
		font-size: var(--num-lg);
		font-weight: var(--weight-num);
		color: var(--text);
	}

	.score.dim {
		color: var(--text-muted);
	}

	.colon {
		font-size: var(--num-md);
		font-weight: var(--weight-num);
		color: var(--accent);
	}

	.time {
		font-size: var(--text-sm);
		font-weight: 700;
		color: var(--text-muted);
	}

	.prob {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-3);
		margin: var(--space-4) 0 0;
		padding-block-start: var(--space-3);
		border-block-start: 1px solid var(--line);
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--text-muted);
	}

	.lg .prob {
		flex-direction: column;
		gap: var(--space-2);
		font-size: var(--text-md);
	}

	.sets {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: center;
		gap: var(--space-1) var(--space-4);
		margin: var(--space-4) 0 0;
		padding-block-start: var(--space-3);
		border-block-start: 1px solid var(--line);
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.strong {
		font-weight: var(--weight-num);
		color: var(--text);
	}

	.golden {
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
	}
</style>
