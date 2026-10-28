<script lang="ts">
	import { page } from '$app/state';
	import { SITE_DESCRIPTION, SITE_URL } from '#lib/site.js';

	let { title, description = SITE_DESCRIPTION }: { title: string; description?: string } = $props();

	// og:url 與 og:image 必須是絕對網址；pathname 已含 base，對著正式站解析即可。
	const url = $derived(new URL(page.url.pathname, SITE_URL).href);
	const image = `${SITE_URL}/og-image.png`;
</script>

<svelte:head>
	<title>{title}</title>
	<meta name="description" content={description} />
	<meta property="og:type" content="website" />
	<meta property="og:title" content={title} />
	<meta property="og:description" content={description} />
	<meta property="og:image" content={image} />
	<meta property="og:url" content={url} />
	<meta name="twitter:card" content="summary_large_image" />
</svelte:head>
