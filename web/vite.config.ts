import { readFileSync } from 'node:fs';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';
import { SHELL } from './src/lib/offline.ts';

const base = '';

export default defineConfig({
	plugins: [
		{
			// vite preview 只提供 .svelte-kit/output 的檔，不含 adapter 寫進 build/ 的空殼頁；少了它，service worker 在預覽與 e2e 裡裝不起來
			name: 'serve-shell-in-preview',
			configurePreviewServer(server) {
				server.middlewares.use(`${base}/${SHELL}`, (_req, res) => {
					res.setHeader('content-type', 'text/html');
					res.end(readFileSync(`build/${SHELL}`));
				});
			}
		},
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter({ fallback: SHELL }),
			// 由頁面自己註冊（#lib/serviceWorker.ts），換版時機才控制得住
			serviceWorker: { register: false },
			// 站台在網域根目錄（正式網址 https://vstat.tw）；dev、preview、正式站一致
			paths: { base, relative: false }
		})
	],
	test: {
		include: ['src/**/*.test.ts', 'e2e/**/*.test.ts'],
		environment: 'node'
	}
});
