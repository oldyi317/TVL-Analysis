<script lang="ts">
	import { resolve } from '$app/paths';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import MatchLink from '#lib/components/MatchLink.svelte';
	import StandingsTable from '#lib/components/StandingsTable.svelte';
	import { stageLabel } from '#lib/match.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const params = $derived({ season: data.season.name, group: data.group });
	const matchHref = (game: number) =>
		`${resolve('/[season]/[group]/matches/[game]', { ...params, game: String(game) })}/`;
</script>

<svelte:head>
	<title>{data.season.name} {GROUP_LABEL[data.group]}｜TVL 數據</title>
</svelte:head>

<h1>{data.season.name} {GROUP_LABEL[data.group]}</h1>

<section aria-labelledby="h-latest">
	<h2 id="h-latest">最新比分</h2>
	{#if data.latest}
		<ol class="cards">
			{#each data.latest.matches as match (match.game_id)}
				<li>
					<MatchLink href={matchHref(match.game_id)} {match} stage={stageLabel(match.stage, match.week)} />
				</li>
			{/each}
		</ol>
		<a class="more" href="{resolve('/[season]/[group]/matches', params)}/">全部賽事</a>
	{:else}
		<EmptyState title="尚無賽事" hint="賽事排定後就會出現" />
	{/if}
</section>

<section aria-labelledby="h-standings">
	<h2 id="h-standings">戰績</h2>
	{#if data.rows.length === 0}
		<EmptyState title="尚無戰績" hint="這一季開打後就會出現" />
	{:else}
		<StandingsTable rows={data.rows.slice(0, 4)} note={data.note ?? undefined} />
		<a class="more" href="{resolve('/[season]/[group]/standings', params)}/">完整戰績</a>
	{/if}
</section>

{#if data.next}
	<section aria-labelledby="h-next">
		<h2 id="h-next">下一場</h2>
		<MatchLink href={matchHref(data.next.game_id)} match={data.next} stage={stageLabel(data.next.stage, data.next.week)} />
	</section>
{/if}

<style>
	h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin-block: 1.5rem 0.75rem;
	}

	h2 {
		font-size: var(--text-lg);
		font-weight: 700;
		margin-block: var(--space-6) var(--space-3);
	}

	.cards {
		display: grid;
		gap: var(--space-3);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.more {
		display: inline-flex;
		align-items: center;
		min-block-size: var(--tap);
		margin-block-start: var(--space-2);
		font-weight: 500;
		text-decoration: underline;
		text-underline-offset: 0.2em;
	}

	.more::after {
		content: ' ›';
	}
</style>
