<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import SiteHead from '#lib/components/SiteHead.svelte';
	import { browserStorage, DEFAULT_GROUP, readGroup } from '#lib/groupMemory.js';
	import { currentSeason, GROUP_LABEL, type Group } from '#lib/site.js';

	const href = (group: Group) =>
		`${resolve('/[season]/[group]', { season: currentSeason.name, group })}/`;

	onMount(() => {
		goto(href(readGroup(browserStorage())), { replaceState: true });
	});
</script>

<SiteHead title="TVL 數據" />

<h1>TVL 數據</h1>
<a class="enter" href={href(DEFAULT_GROUP)}>前往 {currentSeason.name} {GROUP_LABEL[DEFAULT_GROUP]}</a>

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
