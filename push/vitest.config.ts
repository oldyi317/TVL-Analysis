import path from 'node:path';
import { cloudflareTest, readD1Migrations } from '@cloudflare/vitest-plugin';
import { defineConfig } from 'vitest/config';

export default defineConfig(async () => {
	const migrations = await readD1Migrations(path.join(import.meta.dirname, 'migrations'));
	return {
		plugins: [
			cloudflareTest({
				wrangler: { configPath: './wrangler.jsonc' },
				// 寫死測試用的設定，本機 .dev.vars 改了這些值也不影響測試
				miniflare: {
					bindings: {
						TEST_MIGRATIONS: migrations,
						ALLOWED_ORIGIN: 'https://vstat.tw',
						SITE_ROOT: 'https://vstat.tw',
						FEED_URL: 'https://vstat.tw/push/matches.json',
						SEND_WINDOW: '08:00-22:00',
						TODAY_WINDOW: '10:00-11:00'
					}
				}
			})
		],
		test: { setupFiles: ['./test/apply-migrations.ts'] }
	};
});
