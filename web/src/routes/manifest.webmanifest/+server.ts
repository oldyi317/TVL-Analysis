import { json } from '@sveltejs/kit';
import { asset, resolve } from '$app/paths';
import type { RequestHandler } from './$types';

export const prerender = true;

const THEME = '#f3f4f9';

export const GET: RequestHandler = () => {
	const root = resolve('/');
	return json(
		{
			name: 'VStat 企排',
			short_name: 'VStat 企排',
			display: 'standalone',
			start_url: root,
			scope: root,
			id: root,
			background_color: THEME,
			theme_color: THEME,
			icons: [
				{ src: asset('icons/icon-192.png'), sizes: '192x192', type: 'image/png' },
				{ src: asset('icons/icon-512.png'), sizes: '512x512', type: 'image/png' },
				{
					src: asset('icons/icon-maskable-512.png'),
					sizes: '512x512',
					type: 'image/png',
					purpose: 'maskable'
				}
			]
		},
		{ headers: { 'content-type': 'application/manifest+json' } }
	);
};
