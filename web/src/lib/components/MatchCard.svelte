<script lang="ts">
	import { setsWon, winnerSide, winProbPercent, type StyleMatch } from '#lib/match.js';
	import TeamBadge from './TeamBadge.svelte';
	import { teamStyle } from './teamStyle.js';

	let {
		match,
		stage,
		showSets = true
	}: { match: StyleMatch; stage?: string | null; showSets?: boolean } = $props();

	const played = $derived(match.sets.length > 0);
	const winner = $derived(winnerSide(match.sets));
	const [homeSets, awaySets] = $derived(setsWon(match.sets));
	const prob = $derived(winProbPercent(match.home_win_prob));
</script>

<article style={teamStyle(match.home)} aria-label="{match.home.short_name} 對 {match.away.short_name}">
	<p class="meta">
		<span>{match.date} 星期{match.weekday}{match.start_time ? ` ${match.start_time}` : ''} · {match.venue}</span>
		{#if stage}
			<span class="stage">{stage}</span>
		{/if}
	</p>

	<div class="teams">
		{#each [{ side: 'home', team: match.home, won: homeSets }, { side: 'away', team: match.away, won: awaySets }] as row (row.side)}
			<div class="team" class:win={winner === row.side} class:lose={winner !== null && winner !== row.side}>
				<TeamBadge team={row.team} size="sm" />
				<span class="name">{row.team.short_name}</span>
				{#if played}
					<span class="score">{row.won}</span>
				{/if}
			</div>
		{/each}
		{#if !played}
			<span class="pending">未開打</span>
		{/if}
	</div>

	{#if !played}
		<div class="prob" role="group" aria-label="賽前勝率">
			{#if prob}
				{prob[0]}% – {prob[1]}%
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
					黃金決勝局
					<span class:strong={match.golden_set[0] > match.golden_set[1]}>{match.golden_set[0]}</span>-<span class:strong={match.golden_set[1] > match.golden_set[0]}>{match.golden_set[1]}</span>
				</span>
			{/if}
		</p>
	{/if}
</article>

<style>
	article {
		background: var(--surface);
		border: 1px solid var(--line);
		border-inline-start: 4px solid var(--team-fill);
		border-radius: var(--radius-md);
		padding: var(--space-4);
	}

	.meta {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: var(--space-1) var(--space-3);
		margin: 0 0 var(--space-3);
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.stage {
		font-weight: 500;
	}

	.teams {
		position: relative;
		display: grid;
		gap: var(--space-2);
	}

	.team {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		color: var(--text);
	}

	.team.win {
		font-weight: 700;
	}

	.team.lose {
		color: var(--text-muted);
	}

	.team.lose .score {
		font-weight: 400;
	}

	.name {
		flex: 1;
		min-inline-size: 0;
		font-size: var(--text-lg);
	}

	.score {
		font-size: var(--num-lg);
		font-weight: var(--weight-num);
		line-height: 1;
	}

	.pending {
		position: absolute;
		inset-block: 0;
		inset-inline-end: 0;
		display: flex;
		align-items: center;
		font-size: var(--text-lg);
		font-weight: 700;
		color: var(--text-muted);
	}

	.prob {
		margin: var(--space-3) 0 0;
		padding-block-start: var(--space-3);
		border-block-start: 1px solid var(--line);
		font-size: var(--text-md);
		font-weight: 500;
		color: var(--text-muted);
	}

	.sets {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: var(--space-2);
		margin: var(--space-3) 0 0;
		padding-block-start: var(--space-3);
		border-block-start: 1px solid var(--line);
		font-size: min(var(--num-md), 4.8vw);
		line-height: 1.2;
		color: var(--text-muted);
	}

	.strong {
		font-weight: var(--weight-num);
		color: var(--text);
	}

	.golden {
		flex-basis: 100%;
		font-size: var(--text-sm);
		font-weight: 400;
		line-height: 1.5;
	}
</style>
