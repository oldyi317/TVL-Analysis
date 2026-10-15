<script lang="ts">
	import { onMount } from 'svelte';
	import BigStat from '#lib/components/BigStat.svelte';
	import DataNote from '#lib/components/DataNote.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import MatchCard from '#lib/components/MatchCard.svelte';
	import PlayerCard from '#lib/components/PlayerCard.svelte';
	import SiteHeader from '#lib/components/SiteHeader.svelte';
	import StandingsTable from '#lib/components/StandingsTable.svelte';
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import { teamStyle } from '#lib/components/teamStyle.js';
	import Chart from '#lib/chart/Chart.svelte';
	import { contrastRatio, SURFACE, teamPalette } from '#lib/color.js';
	import { GROUP_LABEL, teams, teamsIn } from '#lib/site.js';
	import {
		demoSeason,
		menStandings,
		sampleMatches,
		samplePlayers,
		trendOption,
		womenStandings2024
	} from '#lib/styleguide/fixtures.js';

	const NEUTRALS = ['--bg', '--surface', '--text', '--text-muted', '--line', '--accent', '--on-accent'];
	const SEMANTICS = ['--win', '--loss', '--warn', '--warn-tint', '--warn-ink'];
	const TYPE_SCALE = [
		{ token: '--text-sm', px: 14, sample: '日期、場館、欄位名' },
		{ token: '--text-md', px: 16, sample: '正文、列表、表格文字' },
		{ token: '--text-lg', px: 18, sample: '區塊標題、隊名' },
		{ token: '--text-xl', px: 22, sample: '頁標題' }
	];
	const NUM_SCALE = [
		{ token: '--num-md', px: 24, sample: '25-23' },
		{ token: '--num-lg', px: 36, sample: '3' },
		{ token: '--num-xl', px: 52, sample: '512' }
	];
	const SPACES = [1, 2, 3, 4, 5, 6, 7];
	const RADII = ['--radius-sm', '--radius-md', '--radius-full'];

	const menTeams = teamsIn(demoSeason.cup_id, 'men');

	let dark = $state(false);
	let hex = $state<Record<string, string>>({});

	function readTokens() {
		const style = getComputedStyle(document.documentElement);
		hex = Object.fromEntries(
			[...NEUTRALS, ...SEMANTICS].map((t) => [t, style.getPropertyValue(t).trim()])
		);
	}

	onMount(() => {
		const query = matchMedia('(prefers-color-scheme: dark)');
		const sync = () => {
			dark = query.matches;
			readTokens();
		};
		sync();
		query.addEventListener('change', sync);
		return () => query.removeEventListener('change', sync);
	});

	const surface = $derived(dark ? SURFACE.dark : SURFACE.light);

	const barTeams = menTeams.slice(0, 4);
	const barOption = $derived({
		legend: { show: false },
		tooltip: { trigger: 'axis' },
		xAxis: { type: 'category', data: barTeams.map((t) => t.short_name) },
		yAxis: { type: 'value' },
		series: [
			{
				name: '場均得分',
				type: 'bar',
				data: barTeams.map((t, i) => {
					const p = teamPalette(t.color);
					return { value: [92, 88, 85, 80][i], itemStyle: { color: dark ? p.inkDark : p.inkLight } };
				})
			}
		]
	});
</script>

<svelte:head>
	<title>樣式指南｜TVL 數據</title>
</svelte:head>

<h1>樣式指南</h1>
<p class="lead">本頁把設計 token 與元件用真實球隊資料擺出來，供簽核用。球隊是真的；比分、戰績、球員與圖表數字都是示意資料，不是真實結果。</p>

<section id="colors">
	<h2>色彩</h2>
	<h3>中性色</h3>
	<ul class="swatches">
		{#each NEUTRALS as token (token)}
			<li><span class="chip" style:background="var({token})"></span><code>{token}</code> <span class="hex">{hex[token] ?? ''}</span></li>
		{/each}
	</ul>
	<h3>語意色</h3>
	<ul class="swatches">
		{#each SEMANTICS as token (token)}
			<li><span class="chip" style:background="var({token})"></span><code>{token}</code> <span class="hex">{hex[token] ?? ''}</span></li>
		{/each}
	</ul>
	<h3>隊色</h3>
	<ul class="team-colors">
		{#each teams as team (`${team.group}-${team.team_id}`)}
			{@const p = teamPalette(team.color)}
			{@const ink = dark ? p.inkDark : p.inkLight}
			<li style={teamStyle(team)}>
				<TeamBadge {team} />
				<div>
					<p class="team-name">{team.name}</p>
					<p class="meta">{GROUP_LABEL[team.group]} · fill {p.fill}</p>
					<p class="meta">文字色 {ink} · 對比 {contrastRatio(ink, surface).toFixed(1)}</p>
				</div>
			</li>
		{/each}
	</ul>
</section>

<section>
	<h2>字級</h2>
	<ul class="type">
		{#each TYPE_SCALE as step (step.token)}
			<li>
				<code>{step.token} · {step.px}px</code>
				<span style:font-size="var({step.token})">{step.sample}</span>
			</li>
		{/each}
		{#each NUM_SCALE as step (step.token)}
			<li>
				<code>{step.token} · {step.px}px</code>
				<span class="num" style:font-size="var({step.token})">{step.sample}</span>
			</li>
		{/each}
	</ul>
</section>

<section>
	<h2>間距與圓角</h2>
	<ul class="spaces">
		{#each SPACES as n (n)}
			<li><code>--space-{n}</code><span class="bar" style:inline-size="var(--space-{n})"></span></li>
		{/each}
	</ul>
	<ul class="radii">
		{#each RADII as token (token)}
			<li><span class="box" style:border-radius="var({token})"></span><code>{token}</code></li>
		{/each}
	</ul>
</section>

<section>
	<h2>頁首</h2>
	<div class="frame">
		<SiteHeader season={demoSeason.name} group="men" hrefFor={() => '#'} />
	</div>
</section>

<section>
	<h2>賽事卡</h2>
	<div class="stack">
		{#each sampleMatches as match (match.date)}
			<MatchCard {match} />
		{/each}
	</div>
</section>

<section>
	<h2>戰績表列</h2>
	<h3>{demoSeason.name} {GROUP_LABEL.men}</h3>
	<StandingsTable rows={menStandings} />
	<h3>2024-25 {GROUP_LABEL.women}</h3>
	<StandingsTable
		rows={womenStandings2024}
		note="2024-25 女子組官網只收錄 8 場賽事，戰績不完整"
	/>
</section>

<section>
	<h2>隊標記</h2>
	<ul class="badges">
		{#each ['sm', 'md', 'lg'] as const as size (size)}
			<li>
				{#each menTeams.slice(0, 3) as team (team.team_id)}
					<TeamBadge {team} {size} />
				{/each}
				<code>{size}</code>
			</li>
		{/each}
	</ul>
</section>

<section>
	<h2>球員卡</h2>
	<div class="stack">
		{#each samplePlayers as player (player.name)}
			<PlayerCard {player} />
		{/each}
	</div>
</section>

<section>
	<h2>大數字</h2>
	<div class="stats">
		<BigStat label="攻擊得分" value={184} />
		<BigStat label="攔網" value={31} />
		<BigStat label="發球" value={12} />
	</div>
</section>

<section>
	<h2>空資料與註記</h2>
	<div class="stack">
		<EmptyState title="這一季還沒有賽事" hint="開賽後會自動出現在這裡" />
		<DataNote text="2024-25 女子組官網只收錄 8 場賽事，戰績不完整" />
	</div>
</section>

<section>
	<h2>圖表</h2>
	<Chart option={trendOption} ariaLabel="單場得分趨勢折線圖" />
	<Chart option={barOption} ariaLabel="四隊場均得分比較長條圖" />
</section>

<style>
	h1 {
		margin-block: var(--space-5) var(--space-2);
	}

	.lead {
		margin: 0 0 var(--space-6);
		color: var(--text-muted);
	}

	section {
		margin-block-end: var(--space-6);
	}

	h2 {
		margin-block-end: var(--space-3);
	}

	h3 {
		margin-block: var(--space-4) var(--space-2);
	}

	ul {
		list-style: none;
		margin: 0;
		padding: 0;
	}

	code {
		font-size: var(--text-sm);
	}

	.swatches {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(10rem, 1fr));
		gap: var(--space-3);
	}

	.swatches li {
		display: grid;
		gap: var(--space-1);
	}

	.chip {
		block-size: 2.5rem;
		border: 1px solid var(--line);
		border-radius: var(--radius-sm);
	}

	.hex {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.team-colors {
		display: grid;
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: var(--radius-md);
	}

	.team-colors li {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		padding: var(--space-3) var(--space-4);
	}

	.team-colors li + li {
		border-block-start: 1px solid var(--line);
	}

	p {
		margin: 0;
	}

	.team-name {
		font-weight: 700;
		color: var(--team-ink);
	}

	.meta {
		font-size: var(--text-sm);
		color: var(--text-muted);
	}

	.type li {
		display: grid;
		padding-block: var(--space-2);
		border-block-end: 1px solid var(--line);
	}

	.num {
		font-weight: var(--weight-num);
		line-height: 1.2;
	}

	.spaces li {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		margin-block-end: var(--space-1);
	}

	.spaces code {
		inline-size: 6rem;
	}

	.bar {
		block-size: 1rem;
		background: var(--text);
	}

	.radii {
		display: flex;
		gap: var(--space-4);
		margin-block-start: var(--space-4);
	}

	.radii li {
		display: grid;
		justify-items: center;
		gap: var(--space-1);
	}

	.box {
		inline-size: 4rem;
		block-size: 4rem;
		background: var(--surface);
		border: 2px solid var(--text);
	}

	.frame {
		border: 1px solid var(--line);
	}

	.stack {
		display: grid;
		gap: var(--space-3);
	}

	.badges li {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		margin-block-end: var(--space-3);
	}

	.stats {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: var(--space-3);
	}
</style>
