<script lang="ts">
	import { onMount } from 'svelte';
	import BigStat from '#lib/components/BigStat.svelte';
	import DataNote from '#lib/components/DataNote.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import Icon from '#lib/components/Icon.svelte';
	import MatchCard from '#lib/components/MatchCard.svelte';
	import PlayerCard from '#lib/components/PlayerCard.svelte';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import StandingsTable from '#lib/components/StandingsTable.svelte';
	import TeamBadge from '#lib/components/TeamBadge.svelte';
	import WinRing from '#lib/components/WinRing.svelte';
	import { teamStyle } from '#lib/components/teamStyle.js';
	import Chart from '#lib/chart/Chart.svelte';
	import { contrastRatio, SURFACE, teamPalette } from '#lib/color.js';
	import { currentSeason, GROUP_LABEL, seasonTeam, teams, teamsIn } from '#lib/site.js';
	import {
		demoSeason,
		menStandings,
		sampleMatches,
		samplePlayers,
		trendOption,
		womenStandings2024
	} from '#lib/styleguide/fixtures.js';

	const TOKENS = [
		['--bg', '#f3f4f9', '#12141f'],
		['--surface', '#ffffff', '#1b1e2e'],
		['--surface-2', '#eef0f8', '#242838'],
		['--text', '#262b4a', '#eceef8'],
		['--text-muted', '#5f6788', '#9ba1bc'],
		['--line', '#c9cde2', '#3a405a'],
		['--accent', '#4f5bd5', '#7b82f0'],
		['--accent-strong', '#3d48bf', '#959bf5'],
		['--accent-tint', '#e9ebfb', '#262b4a'],
		['--series-1', '#5258da', '#7b82f0'],
		['--series-2', '#e2457f', '#f66fa0'],
		['--series-3', '#cc6a00', '#f8b36a'],
		['--series-4', '#6b72d4', '#5f66b8'],
		['--series-5', '#0b9279', '#3fd2b6'],
		['--win', '#12804f', '#3ed48e'],
		['--loss', '#cc3347', '#f2728a'],
		['--warn-tint', '#fff3c4', '#3a2a0a'],
		['--warn-ink', '#7a5a00', '#fde68a'],
		['--live', '#f0555a', '#f77177']
	] as const;
	const MODES = [
		{ label: '淺色', index: 1 },
		{ label: '深色', index: 2 }
	] as const;
	const TYPE_SCALE = [
		{ token: '--text-sm', px: 14, sample: '日期、場館、欄位名' },
		{ token: '--text-md', px: 16, sample: '正文、列表、表格文字' },
		{ token: '--text-lg', px: 18, sample: '區塊標題、隊名' },
		{ token: '--text-xl', px: 22, sample: '頁標題' },
		{ token: '--text-2xl', px: 28, sample: '桌面頁標題' }
	];
	const NUM_SCALE = [
		{ token: '--num-md', px: 24, sample: '25-23' },
		{ token: '--num-lg', px: 36, sample: '3' },
		{ token: '--num-xl', px: 52, sample: '512' }
	];
	const SPACES = [1, 2, 3, 4, 5, 6, 7];
	const RADII = ['--radius-sm', '--radius-md', '--radius-lg', '--radius-full'];
	const ICONS = [
		'home',
		'calendar',
		'trophy',
		'shield',
		'user',
		'chevron-left',
		'arrow-up-right',
		'map-pin'
	] as const;
	const TAGS = [
		{ cls: 'tag-win', text: '勝' },
		{ cls: 'tag-loss', text: '敗' },
		{ cls: 'tag-pending', text: '未開打' },
		{ cls: 'tag-live', text: '本週' },
		{ cls: 'tag-stage', text: '例行賽・第 3 週' }
	];

	const menTeams = teamsIn(demoSeason.cup_id, 'men');

	let dark = $state(false);

	onMount(() => {
		const query = matchMedia('(prefers-color-scheme: dark)');
		const sync = () => (dark = query.matches);
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

<SiteHead title="樣式指南｜TVL 數據" />

<h1>樣式指南</h1>
<p class="lead">本頁把設計 token 與元件用真實球隊資料擺出來，供簽核用。球隊是真的；比分、戰績、球員與圖表數字都是示意資料，不是真實結果。</p>

<section id="colors">
	<h2>色彩</h2>
	<div class="modes">
		{#each MODES as mode (mode.label)}
			<div class="card">
				<h3>{mode.label}</h3>
				<ul class="swatches">
					{#each TOKENS as token (token[0])}
						<li>
							<span class="swatch" style:background={token[mode.index]}></span>
							<code>{token[0]}</code>
							<span class="hex">{token[mode.index]}</span>
						</li>
					{/each}
				</ul>
			</div>
		{/each}
	</div>
	<h3 class="sub">隊色</h3>
	<ul class="team-colors card">
		{#each teams as team (`${team.group}-${team.team_id}`)}
			{@const p = teamPalette(team.color)}
			{@const ink = dark ? p.inkDark : p.inkLight}
			<li style={teamStyle(team)}>
				<TeamBadge team={seasonTeam(team, currentSeason.cup_id)} />
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
	<ul class="type card">
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
	<div class="card">
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
	</div>
</section>

<section>
	<h2>控制項與標籤</h2>
	<div class="card controls">
		<div>
			<h3>Chip</h3>
			<div class="row">
				<span class="chip" aria-current="true">2025-26</span>
				<span class="chip">2024-25</span>
				<span class="chip">2023-24</span>
			</div>
		</div>
		<div>
			<h3>分段控制</h3>
			<div class="segmented">
				<span class="seg" aria-current="true">男子組</span>
				<span class="seg">女子組</span>
			</div>
		</div>
		<div>
			<h3>標籤</h3>
			<div class="row">
				{#each TAGS as tag (tag.text)}
					<span class="tag {tag.cls}">{tag.text}</span>
				{/each}
			</div>
		</div>
		<div>
			<h3>區塊標題連結</h3>
			<div class="section-head">
				<h2>最新比分</h2>
				<a href="#colors">全部賽事</a>
			</div>
		</div>
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
	<h2>數據磚與甜甜圈</h2>
	<div class="stats">
		<BigStat label="攻擊得分" value={184} tone={1} />
		<BigStat label="攔網" value={31} tone={2} />
		<BigStat label="發球" value={12} tone={3} detail="每局 0.4" />
	</div>
	<div class="card rings">
		<WinRing value={62} size={88} label="主隊勝率" />
		<WinRing value={38} size={56} />
	</div>
</section>

<section>
	<h2>戰績表</h2>
	<h3 class="sub">{demoSeason.name} {GROUP_LABEL.men}</h3>
	<StandingsTable rows={menStandings} />
	<h3 class="sub">2024-25 {GROUP_LABEL.women}</h3>
	<StandingsTable
		rows={womenStandings2024}
		note="2024-25 女子組官網只收錄 8 場賽事，戰績不完整"
	/>
</section>

<section>
	<h2>隊色圓標</h2>
	<ul class="badges card">
		{#each ['sm', 'md', 'lg', 'xl'] as const as size (size)}
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
	<h2>球員列</h2>
	<ul class="players card">
		{#each samplePlayers as player (player.name)}
			<li><PlayerCard {player} /></li>
		{/each}
	</ul>
</section>

<section>
	<h2>空資料與註記</h2>
	<div class="stack">
		<div class="card"><EmptyState title="這一季還沒有賽事" hint="開賽後會自動出現在這裡" /></div>
		<DataNote text="2024-25 女子組官網只收錄 8 場賽事，戰績不完整" />
	</div>
</section>

<section>
	<h2>Icon</h2>
	<ul class="icons card">
		{#each ICONS as name (name)}
			<li><Icon {name} size={24} /><code>{name}</code></li>
		{/each}
	</ul>
</section>

<section>
	<h2>圖表</h2>
	<div class="stack">
		<div class="card"><Chart option={trendOption} ariaLabel="單場得分趨勢折線圖" /></div>
		<div class="card"><Chart option={barOption} ariaLabel="四隊場均得分比較長條圖" /></div>
	</div>
</section>

<style>
	h1 {
		margin-block: var(--space-4) var(--space-2);
	}

	.lead {
		margin: 0 0 var(--space-5);
		color: var(--text-muted);
	}

	section {
		margin-block-end: var(--space-6);
	}

	section > h2 {
		margin-block-end: var(--space-3);
	}

	h3 {
		margin-block-end: var(--space-2);
	}

	h3.sub {
		margin-block: var(--space-4) var(--space-2);
	}

	ul {
		list-style: none;
		margin: 0;
		padding: 0;
	}

	p {
		margin: 0;
	}

	code {
		font-size: var(--text-sm);
	}

	.modes {
		display: grid;
		gap: var(--space-3);
	}

	@media (min-width: 64rem) {
		.modes {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}

	.swatches {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(9rem, 1fr));
		gap: var(--space-3);
	}

	.swatches li {
		display: grid;
		gap: var(--space-1);
	}

	.swatch {
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
		padding-block: var(--space-2);
	}

	.team-colors li {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		padding: var(--space-3) 0;
	}

	.team-colors li + li {
		border-block-start: 1px solid var(--line);
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
	}

	.type li + li {
		border-block-start: 1px solid var(--line);
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
		border-radius: var(--radius-full);
		background: var(--accent);
	}

	.radii {
		display: flex;
		flex-wrap: wrap;
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
		background: var(--accent-tint);
		border: 2px solid var(--accent);
	}

	.controls {
		display: grid;
		gap: var(--space-4);
	}

	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-2);
	}

	.segmented {
		display: flex;
		gap: 4px;
		padding: 4px;
		border-radius: var(--radius-full);
		background: var(--surface-2);
		max-inline-size: 22rem;
	}

	.seg {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		min-block-size: var(--tap);
		border-radius: var(--radius-full);
		color: var(--text-muted);
		font-weight: 700;
	}

	.seg[aria-current] {
		background: var(--surface);
		color: var(--accent);
		box-shadow: var(--shadow-card);
		outline: 1px solid var(--card-edge);
	}

	.stack {
		display: grid;
		gap: var(--space-3);
	}

	.stats {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: var(--space-2);
		margin-block-end: var(--space-3);
	}

	.rings {
		display: flex;
		align-items: center;
		gap: var(--space-5);
	}

	.badges li {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		margin-block-end: var(--space-3);
	}

	.players {
		padding: var(--space-2);
	}

	.icons {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(8rem, 1fr));
		gap: var(--space-4);
		color: var(--accent);
	}

	.icons li {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}

	.icons code {
		color: var(--text-muted);
	}
</style>
