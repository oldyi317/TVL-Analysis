import type { Reporter, TestCase, TestResult } from '@playwright/test/reporter';

// 預設 list reporter 不印 skip 的原因；這裡補印，讓每條 skipped 看得到 reason
export default class SkipReasonReporter implements Reporter {
	onTestEnd(test: TestCase, result: TestResult) {
		if (result.status !== 'skipped') return;
		const reason = test.annotations.find((a) => a.type === 'skip')?.description ?? '（未寫原因）';
		console.log(`  跳過：${test.title} -- ${reason}`);
	}
}
