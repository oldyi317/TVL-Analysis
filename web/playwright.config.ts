import { defineConfig, devices } from '@playwright/test';

// 以手機視窗對建置產物（vite preview）跑 smoke
export default defineConfig({
	testDir: 'e2e',
	testMatch: '**/*.e2e.{ts,js}',
	use: { ...devices['Pixel 7'], baseURL: 'http://localhost:4173' },
	// 先 npm run build 再跑測試（test:e2e 已串好）；/mnt/d 上 node 啟動很慢，等待放寬到 3 分鐘
	webServer: {
		command: 'npm run preview',
		port: 4173,
		timeout: 180_000,
		reuseExistingServer: !process.env.CI
	}
});
