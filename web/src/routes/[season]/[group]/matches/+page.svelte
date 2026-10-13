<script lang="ts">
	import { resolve } from '$app/paths';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import MatchCard from '#lib/components/MatchCard.svelte';
	import { stageLabel } from '#lib/match.js';
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

	const matchHref = (game: number) =>
		`${resolve('/[season]/[group]/matches/[game]', {
			season: data.season.name,
			group: data.group,
			game: String(game)
		})}/`;
</script>

<svelte:head>
	<title>{data.season.name} {GROUP_LABEL[data.group]} 賽事｜TVL 數據</title>
</svelte:head>

<h1>{data.season.name} {GROUP_LABEL[data.group]} 賽事</h1>

{#if data.months.length === 0}
	<EmptyState title="尚無賽事" hint="賽事排定後就會出現" />
{:else}
	<nav class="months" aria-label="月份">
		{#each data.months as month (month.key)}
			<a
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
				{#each month.matches as match (match.game_id)}
					<li>
						<a class="card-link" href={matchHref(match.game_id)}>
							<MatchCard {match} stage={stageLabel(match.stage, match.week)} />
						</a>
					</li>
				{/each}
			</ol>
		</section>
	{/each}
{/if}

<style>
	h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin-block: 1.5rem 0.75rem;
	}

	.months {
		position: sticky;
		inset-block-start: 0;
		z-index: 1;
		display: flex;
		gap: var(--space-2);
		margin-inline: calc(-1 * var(--space-4));
		padding: var(--space-2) var(--space-4);
		overflow-x: auto;
		scrollbar-width: none;
		background: var(--bg);
		border-block-end: 1px solid var(--line);
	}

	.months::-webkit-scrollbar {
		display: none;
	}

	.months a {
		display: inline-flex;
		align-items: center;
		flex: none;
		min-block-size: var(--tap);
		padding-inline: var(--space-3);
		font-size: var(--text-md);
		color: var(--text-muted);
		border: 1px solid transparent;
		border-radius: var(--radius-sm);
	}

	.months a[aria-current='location'] {
		color: var(--on-accent);
		background: var(--accent);
		border-color: var(--accent);
	}

	/* 月份列高 = 44px 觸控高 + 上下各 8px 內距 + 1px 線；錨點停在列下方再留 16px */
	section {
		scroll-margin-block-start: calc(var(--tap) + 2 * var(--space-2) + 1px + var(--space-4));
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

	.card-link {
		display: block;
		border-radius: var(--radius-md);
	}

	.card-link:focus-visible {
		outline: 2px solid var(--focus);
		outline-offset: 2px;
	}

	/* 整張卡是連結：指標停留與按下時外框加深；只動上下與右側，左側隊色條不變 */
	@media (hover: hover) {
		.card-link:hover :global(article) {
			border-block-color: var(--text-muted);
			border-inline-end-color: var(--text-muted);
		}
	}

	.card-link:active :global(article) {
		border-block-color: var(--text);
		border-inline-end-color: var(--text);
	}
</style>
