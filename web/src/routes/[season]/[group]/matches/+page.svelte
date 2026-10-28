<script lang="ts">
	import { resolve } from '$app/paths';
	import DataNote from '#lib/components/DataNote.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import MatchLink from '#lib/components/MatchLink.svelte';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { isPlayed, stageLabel, WIN_PROB_NOTE } from '#lib/match.js';
	import { GROUP_LABEL } from '#lib/site.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	let current = $state<string | undefined>(undefined);
	let sections: HTMLElement[] = $state([]);

	// 捲動時標記目前看到的月份：最後一個頂端已到達錨點停靠位置（scroll-margin）的段落；捲到底就是最後一個月
	function track() {
		if (sections.length === 0) return;
		const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 1;
		if (atBottom) {
			current = data.months.at(-1)?.key;
			return;
		}
		const threshold = parseFloat(getComputedStyle(sections[0]).scrollMarginBlockStart) + 1;
		let seen = data.months[0]?.key;
		sections.forEach((section, i) => {
			if (section.getBoundingClientRect().top <= threshold) seen = data.months[i]?.key;
		});
		current = seen;
	}

	$effect(() => {
		let ticking = false;
		const onScroll = () => {
			if (ticking) return;
			ticking = true;
			requestAnimationFrame(() => {
				track();
				ticking = false;
			});
		};
		track();
		window.addEventListener('scroll', onScroll, { passive: true });
		return () => window.removeEventListener('scroll', onScroll);
	});

	const hasUnplayed = $derived(data.months.some((month) => month.matches.some((m) => !isPlayed(m))));

	const matchHref = (game: number) =>
		`${resolve('/[season]/[group]/matches/[game]', {
			season: data.season.name,
			group: data.group,
			game: String(game)
		})}/`;
</script>

<SiteHead title="{data.season.name} {GROUP_LABEL[data.group]} 賽事｜TVL 數據" />

<h1>{data.season.name} {GROUP_LABEL[data.group]} 賽事</h1>

{#if data.months.length === 0}
	<EmptyState title="尚無賽事" hint="賽事排定後就會出現" />
{:else}
	<nav class="months" aria-label="月份">
		{#each data.months as month (month.key)}
			<a
				class="chip"
				href="#m-{month.key}"
				aria-current={current === month.key ? 'location' : undefined}
				onclick={() => (current = month.key)}
			>
				{month.short}
			</a>
		{/each}
	</nav>

	{#each data.months as month, i (month.key)}
		<section id="m-{month.key}" aria-labelledby="h-{month.key}" bind:this={sections[i]}>
			<h2 id="h-{month.key}">{month.label}</h2>
			<ol class="cards">
				{#each month.matches as match, j (match.game_id)}
					<li class:rise={i === 0 && j < 6} style:--i={j + 1}>
						<MatchLink href={matchHref(match.game_id)} {match} stage={stageLabel(match.stage, match.week)} />
					</li>
				{/each}
			</ol>
		</section>
	{/each}
	{#if hasUnplayed}
		<div class="note"><DataNote text={WIN_PROB_NOTE} /></div>
	{/if}
{/if}

<style>
	h1 {
		margin-block: var(--space-4);
	}

	.months {
		position: sticky;
		inset-block-start: 0;
		z-index: 1;
		display: flex;
		gap: var(--space-1);
		margin-inline: calc(-1 * var(--space-4));
		padding: var(--space-2) var(--space-4);
		overflow-x: auto;
		scrollbar-width: none;
		background: var(--bg);
	}

	.months::-webkit-scrollbar {
		display: none;
	}

	.months a {
		flex: none;
	}

	/* 月份列高 = 44px 觸控高 + 上下各 8px 內距；錨點停在列下方再留 16px */
	section {
		scroll-margin-block-start: calc(var(--tap) + 2 * var(--space-2) + var(--space-4));
	}

	h2 {
		margin-block: var(--space-6) var(--space-3);
	}

	.note {
		margin-block-start: var(--space-6);
	}

	.cards {
		display: grid;
		gap: var(--space-3);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	@media (min-width: 64rem) {
		.cards {
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: var(--space-4);
		}

		.months {
			margin-inline: 0;
			padding-inline: 0;
		}
	}
</style>
