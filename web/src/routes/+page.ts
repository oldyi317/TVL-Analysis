import { redirect } from '@sveltejs/kit';
import { resolve } from '$app/paths';
import { currentSeason } from '#lib/site.js';

export function load(): never {
	// 結尾斜線與 trailingSlash='always' 一致，避免 GitHub Pages 多一次 301
	redirect(307, `${resolve('/[season]/[group]', { season: currentSeason.name, group: 'men' })}/`);
}
