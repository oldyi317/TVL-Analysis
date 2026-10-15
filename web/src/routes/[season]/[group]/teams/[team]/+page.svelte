<script lang="ts">
	import { resolve } from '$app/paths';
	import BigStat from '#lib/components/BigStat.svelte';
	import DataNote from '#lib/components/DataNote.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import MatchLink from '#lib/components/MatchLink.svelte';
	import PlayerLink from '#lib/components/PlayerLink.svelte';
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import { stageLabel, WIN_PROB_NOTE } from '#lib/match.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const team = $derived(data.team);
	const listHref = $derived(
		`${resolve('/[season]/[group]/teams', { season: data.season.name, group: data.group })}/`
	);

	const hasUnplayed = $derived(data.matches.some((m) => m.sets.length === 0));

	function matchHref(gameId: number): string {
		return `${resolve('/[season]/[group]/matches/[game]', {
			season: data.season.name,
			group: data.group,
			game: String(gameId)
		})}/`;
	}
</script>

<svelte:head>
	<title>{team.name}｜{data.season.name} {GROUP_LABEL[data.group]} 球隊｜TVL 數據</title>
</svelte:head>

<a class="back" href={listHref}>
	<svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
		<path d="M10 3 5 8l5 5" />
	</svg>
	球隊列表
</a>

<header>
	<TeamBadge {team} size="lg" />
	<h1>{team.name}</h1>
</header>

<h2>戰績</h2>
{#if data.standing}
	<div class="stats">
		<BigStat label="名次" value={data.standing.rank} />
		<BigStat label="勝" value={data.standing.wins} />
		<BigStat label="敗" value={data.standing.losses} />
		<BigStat label="積分" value={data.standing.points} />
	</div>
{:else}
	<EmptyState title="本季尚無戰績" />
{/if}

<h2>賽事</h2>
{#if data.matches.length === 0}
	<EmptyState title="尚無賽事" />
{:else}
	<ol class="cards">
		{#each data.matches as match (match.game_id)}
			<li>
				<MatchLink href={matchHref(match.game_id)} {match} stage={stageLabel(match.stage, match.week)} />
			</li>
		{/each}
	</ol>
	{#if hasUnplayed}
		<div class="note"><DataNote text={WIN_PROB_NOTE} /></div>
	{/if}
{/if}

<h2>週登錄</h2>
{#if data.roster}
	<p class="week">
		{data.roster.week_label}{data.roster.week_start_date ? `（${data.roster.week_start_date}）` : ''}
	</p>
	<ul class="players">
		{#each data.roster.players as player (player.player_id)}
			<li>
				<PlayerLink season={data.season.name} group={data.group} player={{ ...player, team }} />
			</li>
		{/each}
	</ul>
{:else}
	<EmptyState title="這一季沒有週登錄資料" hint="有登錄資料後就會出現" />
{/if}

<style>
	.note {
		margin-block: var(--space-3) 0;
	}

	.back {
		display: inline-flex;
		align-items: center;
		gap: var(--space-1);
		min-block-size: var(--tap);
		margin-block-start: var(--space-2);
		font-weight: 500;
	}

	.back svg {
		flex: none;
	}

	header {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		margin-block-end: var(--space-3);
	}

	h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin: 0;
	}

	h2 {
		font-size: var(--text-lg);
		font-weight: 700;
		margin-block: var(--space-6) var(--space-3);
	}

	.stats {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: var(--space-2);
		padding: var(--space-3);
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: var(--radius-md);
	}

	.cards,
	.players {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: var(--space-3);
	}

	.players {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.week {
		margin: 0 0 var(--space-3);
		color: var(--text-muted);
	}
</style>
