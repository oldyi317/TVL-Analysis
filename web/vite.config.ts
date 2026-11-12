import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter(),
			// 站台在網域根目錄（正式網址 https://vstat.tw）；dev、preview、正式站一致
			paths: { base: '', relative: false }
		})
	],
	test: {
		include: ['src/**/*.test.ts', 'e2e/**/*.test.ts'],
		environment: 'node'
	}
});
