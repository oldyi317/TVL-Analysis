<script lang="ts">
	import { resolve } from '$app/paths';
	import DataNote from '#lib/components/DataNote.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import MatchLink from '#lib/components/MatchLink.svelte';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import StandingsTable from '#lib/components/StandingsTable.svelte';
	import { stageLabel, WIN_PROB_NOTE } from '#lib/match.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const params = $derived({ season: data.season.name, group: data.group });
	const matchHref = (game: number) =>
		`${resolve('/[season]/[group]/matches/[game]', { ...params, game: String(game) })}/`;
</script>

<SiteHead title="{data.season.name} {GROUP_LABEL[data.group]}｜TVL 數據" />

<h1>{data.season.name} {GROUP_LABEL[data.group]}</h1>

<div class="home" class:empty-latest={!data.latest}>
	<section class="latest" aria-labelledby="h-latest">
		<div class="section-head">
			<h2 id="h-latest">最新比分</h2>
			{#if data.latest}
				<a href="{resolve('/[season]/[group]/matches', params)}/">全部賽事</a>
			{/if}
		</div>
		{#if data.latest}
			<ol class="cards">
				{#each data.latest.matches as match, i (match.game_id)}
					<li class="rise" style="--i:{Math.min(i + 1, 6)}">
						<MatchLink href={matchHref(match.game_id)} {match} stage={stageLabel(match.stage, match.week)} />
					</li>
				{/each}
			</ol>
		{:else}
			<EmptyState title="尚無賽事" hint="賽事排定後就會出現" />
		{/if}
	</section>

	<section class="standings" aria-labelledby="h-standings">
		<div class="section-head">
			<h2 id="h-standings">戰績</h2>
			{#if data.rows.length > 0}
				<a href="{resolve('/[season]/[group]/standings', params)}/">完整戰績</a>
			{/if}
		</div>
		{#if data.rows.length === 0}
			<EmptyState title="尚無戰績" hint="這一季開打後就會出現" />
		{:else}
			<div class="rise" style="--i:{Math.min((data.latest?.matches.length ?? 0) + 1, 6)}">
				<StandingsTable rows={data.rows.slice(0, 4)} note={data.note ?? undefined} />
			</div>
		{/if}
	</section>

	{#if data.next}
		<section class="next" aria-labelledby="h-next">
			<div class="section-head">
				<h2 id="h-next">下一場</h2>
			</div>
			<div class="rise" style="--i:{Math.min((data.latest?.matches.length ?? 0) + 2, 6)}">
				<MatchLink href={matchHref(data.next.game_id)} match={data.next} stage={stageLabel(data.next.stage, data.next.week)} size="lg" />
			</div>
			<div class="note"><DataNote text={WIN_PROB_NOTE} /></div>
		</section>
	{/if}
</div>

<style>
	h1 {
		margin-block: var(--space-4) var(--space-4);
	}

	.home {
		display: grid;
		gap: var(--space-6);
	}

	.cards {
		display: grid;
		gap: var(--space-3);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.note {
		margin-block-start: var(--space-3);
	}

	.empty-latest .next {
		order: -1;
	}

	.empty-latest .latest {
		order: 1;
	}

	@media (min-width: 64rem) {
		.home {
			grid-template-columns: minmax(0, 1fr) 22rem;
			grid-template-areas:
				'latest standings'
				'next standings';
			grid-template-rows: auto 1fr;
			align-items: start;
			column-gap: var(--space-6);
		}

		.empty-latest {
			grid-template-areas:
				'next standings'
				'latest standings';
		}

		.latest {
			grid-area: latest;
		}

		.standings {
			grid-area: standings;
		}

		.next {
			grid-area: next;
		}
	}
</style>
