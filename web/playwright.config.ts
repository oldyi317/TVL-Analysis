import { defineConfig, devices } from '@playwright/test';

// 預設 4173；設了 PW_PORT 就在那個 port 起自己的 preview，不沿用別人的（scripts/e2e.sh 用它避開其他 worktree）
const port = Number(process.env.PW_PORT || 4173);

// 以手機視窗對建置產物（vite preview）跑 smoke
export default defineConfig({
	testDir: 'e2e',
	testMatch: '**/*.e2e.{ts,js}',
	// 同一檔的測試也分給不同 worker，不讓最大的檔（leaders）拖成整輪最長的一條；每個測試各有自己的瀏覽器 context，不共用狀態
	fullyParallel: true,
	reporter: [['list'], ['./e2e/skip-reporter.ts']],
	use: { ...devices['Pixel 7'], baseURL: `http://localhost:${port}` },
	// 先 npm run build 再跑測試（test:e2e 已串好）；/mnt/d 上 node 啟動很慢，等待放寬到 3 分鐘
	webServer: {
		command: `npm run preview -- --port ${port} --strictPort`,
		port,
		timeout: 180_000,
		reuseExistingServer: !process.env.CI && !process.env.PW_PORT
	}
});
