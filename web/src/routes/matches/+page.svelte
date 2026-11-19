<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { browserStorage, DEFAULT_GROUP, readGroup } from '#lib/groupMemory.js';
	import { currentSeason, GROUP_LABEL, type Group } from '#lib/site.js';

	// 通知點下去開這裡：轉到這台裝置上次看的組別的當季賽事頁，錨點（比賽日）原樣帶過去
	const href = (group: Group) =>
		`${resolve('/[season]/[group]/matches', { season: currentSeason.name, group })}/`;

	onMount(() => {
		goto(`${href(readGroup(browserStorage()))}${location.hash}`, { replaceState: true });
	});
</script>

<SiteHead title="賽事｜VStat 企排" />

<h1>賽事</h1>
<a class="enter" href={href(DEFAULT_GROUP)}>前往 {currentSeason.name} {GROUP_LABEL[DEFAULT_GROUP]} 賽事</a>

<style>
	h1 {
		font-size: var(--text-xl);
		font-weight: 700;
		margin-block: 1.5rem 0.75rem;
	}

	.enter {
		display: inline-flex;
		align-items: center;
		min-block-size: var(--tap);
		font-weight: 500;
	}
</style>
