<script lang="ts">
	type Name =
		| 'home'
		| 'calendar'
		| 'trophy'
		| 'shield'
		| 'user'
		| 'chevron-left'
		| 'arrow-up-right'
		| 'map-pin';

	let { name, size = 20 }: { name: Name; size?: number } = $props();
</script>

<svg
	width={size}
	height={size}
	viewBox="0 0 24 24"
	fill="none"
	stroke="currentColor"
	stroke-width="1.75"
	stroke-linecap="round"
	stroke-linejoin="round"
	aria-hidden="true"
>
	{#if name === 'home'}
		<path d="M4 11 12 4l8 7" />
		<path d="M6 9.5V19a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V9.5" />
		<path d="M10 20v-5.5a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1V20" />
	{:else if name === 'calendar'}
		<rect x="4" y="5" width="16" height="15" rx="3" />
		<path d="M4 10h16" />
		<path d="M8 3v4" />
		<path d="M16 3v4" />
	{:else if name === 'trophy'}
		<path d="M8 4h8v6a4 4 0 0 1-8 0V4Z" />
		<path d="M8 6H5a1 1 0 0 0-1 1 4 4 0 0 0 4 4" />
		<path d="M16 6h3a1 1 0 0 1 1 1 4 4 0 0 1-4 4" />
		<path d="M12 14v3" />
		<path d="M8.5 20h7" />
		<path d="M10 17h4l.5 3h-5l.5-3Z" />
	{:else if name === 'shield'}
		<path d="M12 3 5 6v5.5c0 4.2 2.9 7.3 7 9.5 4.1-2.2 7-5.3 7-9.5V6l-7-3Z" />
		<path d="m9 12 2 2 4-4" />
	{:else if name === 'user'}
		<circle cx="12" cy="8" r="4" />
		<path d="M4.5 20a7.5 7.5 0 0 1 15 0" />
	{:else if name === 'chevron-left'}
		<path d="m15 5-7 7 7 7" />
	{:else if name === 'arrow-up-right'}
		<path d="M7 17 17 7" />
		<path d="M8 7h9v9" />
	{:else if name === 'map-pin'}
		<path d="M12 21s-6.5-5.7-6.5-11a6.5 6.5 0 0 1 13 0c0 5.3-6.5 11-6.5 11Z" />
		<circle cx="12" cy="10" r="2.5" />
	{/if}
</svg>
