<script lang="ts">
	import { resolve } from '$app/paths';
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import BigStat from '#lib/components/BigStat.svelte';
	import DataNote from '#lib/components/DataNote.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import Icon from '#lib/components/Icon.svelte';
	import MatchLink from '#lib/components/MatchLink.svelte';
	import PlayerLink from '#lib/components/PlayerLink.svelte';
	import { isPlayed, stageLabel, WIN_PROB_NOTE } from '#lib/match.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const CIRC = 2 * Math.PI * 34;

	const team = $derived(data.team);
	const listHref = $derived(
		`${resolve('/[season]/[group]/teams', { season: data.season.name, group: data.group })}/`
	);

	const hasUnplayed = $derived(data.matches.some((m) => !isPlayed(m)));

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
	<Icon name="chevron-left" size={16} />
	球隊列表
</a>

<header class="rise">
	<TeamBadge team={data.seasonTeam} size="xl" label="{data.seasonTeam.name}隊徽" />
	<div class="title">
		<h1>{team.name}</h1>
		<p class="sub">{team.short_name === team.name ? '' : `${team.short_name}・`}{GROUP_LABEL[data.group]}</p>
		{#if data.formerNames.length > 0}
			<p class="former">
				曾用名：{data.formerNames.map((f) => `${f.name}（${f.seasons.join('、')}）`).join('；')}
			</p>
		{/if}
	</div>
</header>

<h2>戰績</h2>
{#if data.standing}
	<div class="stats rise" style:--i="1">
		<div class="rank card">
			<span class="label">名次</span>
			<span class="num">{data.standing.rank}</span>
		</div>
		<div class="stat-tile" style:--tile="var(--tile-1)">
			<span class="label">勝</span>
			<span class="value">{data.standing.wins}</span>
		</div>
		<div class="stat-tile" style:--tile="var(--tile-2)">
			<span class="label">敗</span>
			<span class="value">{data.standing.losses}</span>
		</div>
		<div class="stat-tile" style:--tile="var(--tile-3)">
			<span class="label">積分</span>
			<span class="value">{data.standing.points}</span>
		</div>
		{#if data.standing.win_pct !== null && data.standing.win_pct !== undefined}
			{@const pct = Math.round(data.standing.win_pct * 100)}
			<div class="donut card" role="group" aria-label="勝率">
				<svg viewBox="0 0 88 88" width="88" height="88" aria-hidden="true">
					<circle cx="44" cy="44" r="34" fill="none" stroke="var(--accent-tint)" stroke-width="10" />
					<circle
						class="arc"
						cx="44"
						cy="44"
						r="34"
						fill="none"
						stroke="var(--accent)"
						stroke-width="10"
						stroke-linecap="round"
						stroke-dasharray={CIRC}
						style:--target={CIRC * (1 - pct / 100)}
						style:--circ={CIRC}
						transform="rotate(-90 44 44)"
					/>
				</svg>
				<span class="pct"><b>{pct}%</b><span>勝率</span></span>
			</div>
		{/if}
	</div>
{:else}
	<EmptyState title="本季尚無戰績" />
{/if}

<h2>賽事</h2>
{#if data.matches.length === 0}
	<EmptyState title="尚無賽事" />
{:else}
	<ol class="cards">
		{#each data.matches as match, i (match.game_id)}
			<li class="rise" style:--i={Math.min(i, 5)}>
				<MatchLink href={matchHref(match.game_id)} {match} stage={stageLabel(match.stage, match.week)} />
			</li>
		{/each}
	</ol>
	{#if hasUnplayed}
		<div class="note"><DataNote text={WIN_PROB_NOTE} /></div>
	{/if}
{/if}

<h2>{data.roster?.kind === 'season' ? '賽季報名名單' : '週登錄'}</h2>
{#if data.roster}
	<div class="card">
		<p class="week">
			{data.roster.week_label ?? '官網公布的整季報名名單，尚無週登錄資料'}{data.roster.week_start_date
				? `（${data.roster.week_start_date}）`
				: ''}
		</p>
		<ul class="players">
			{#each data.roster.players as player (player.player_id)}
				<li>
					<PlayerLink season={data.season.name} group={data.group} player={{ ...player, team: data.seasonTeam }} />
				</li>
			{/each}
		</ul>
	</div>
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
		color: var(--accent);
		font-weight: 600;
	}

	.back:hover {
		color: var(--accent-strong);
		text-decoration: underline;
	}

	.back :global(svg) {
		flex: none;
	}

	header {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		margin-block: var(--space-2) var(--space-4);
	}

	.title {
		min-inline-size: 0;
	}

	h1 {
		margin: 0;
	}

	.sub,
	.former {
		margin: var(--space-1) 0 0;
		color: var(--text-muted);
	}

	.former {
		font-size: var(--text-sm);
		font-weight: 600;
	}

	h2 {
		margin-block: var(--space-6) var(--space-3);
	}

	.stats {
		display: grid;
		grid-template-columns: repeat(6, minmax(0, 1fr));
		gap: var(--space-3);
	}

	.rank,
	.donut {
		grid-column: span 3;
	}

	.rank {
		order: 1;
	}

	.donut {
		order: 2;
	}

	.stat-tile {
		grid-column: span 2;
		order: 3;
	}

	.rank {
		display: grid;
		align-content: center;
		gap: var(--space-1);
		padding: var(--space-3);
	}

	.rank .label {
		color: var(--text-muted);
		font-size: var(--text-sm);
		font-weight: 600;
	}

	.rank .num {
		font-size: var(--num-lg);
		font-weight: 800;
	}

	.stat-tile {
		padding: var(--space-3);
	}

	.stat-tile .label {
		font-size: var(--text-sm);
		font-weight: 700;
		opacity: 0.9;
	}

	.stat-tile .value {
		font-size: var(--num-md);
		font-weight: 800;
		line-height: 1.1;
	}

	.donut {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-3);
		padding: var(--space-3);
	}

	.donut svg {
		flex: none;
		inline-size: 4rem;
		block-size: 4rem;
	}

	.arc {
		stroke-dashoffset: var(--target);
		animation: fill 600ms ease-out both;
	}

	@keyframes fill {
		from {
			stroke-dashoffset: var(--circ);
		}
		to {
			stroke-dashoffset: var(--target);
		}
	}

	.pct {
		display: grid;
		color: var(--text-muted);
		font-size: var(--text-sm);
	}

	.pct b {
		color: var(--text);
		font-size: var(--num-md);
		font-weight: 800;
		line-height: 1.2;
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

	@media (min-width: 40rem) {
		.stat-tile {
			padding: var(--space-4);
		}

		.stat-tile .value {
			font-size: var(--num-lg);
		}

	}

	@media (min-width: 64rem) {
		.stats {
			grid-template-columns: 7rem repeat(3, minmax(0, 1fr)) 11rem;
		}

		.rank,
		.donut,
		.stat-tile {
			grid-column: auto;
			order: 0;
		}

		.cards {
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: var(--space-4);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.arc {
			animation: none;
		}
	}
</style>
