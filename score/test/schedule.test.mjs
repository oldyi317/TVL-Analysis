import assert from 'node:assert/strict';
import test from 'node:test';
import worker from '../src/index.ts';
import { runMinute } from '../src/schedule.ts';

const env = {
	FULL_TIME: '09:23', SLIM_TIMES: '15:30,18:30,21:15', DRY_RUN: true,
	FEED_URL: 'https://site.invalid/feed.json', GITHUB_REPO: 'example/league',
	GITHUB_WORKFLOW: 'update.yml', GITHUB_TOKEN: 'test-token'
};
const endpoint = 'https://api.github.com/repos/example/league/actions/workflows/update.yml/dispatches';

test('台北 09:23 直接啟動整包並帶入不寫入設定', async () => {
	const calls = [];
	await runMinute(new Date('2026-11-20T01:23:00Z'), env, async (url, options) => {
		calls.push(url);
		assert.equal(options.method, 'POST');
		assert.equal(options.headers.Authorization, 'Bearer test-token');
		assert.deepEqual(JSON.parse(options.body), { ref: 'main', inputs: { mode: 'full', dry_run: true } });
		return new Response(null, { status: 204 });
	});
	assert.deepEqual(calls, [endpoint]);
});

function feed(date) {
	return { version: 1, cup_id: 22, generated_at: '2026-11-20T09:00:00+08:00',
		links: { men: '/men/', women: '/women/', both: '/matches/' },
		matches: [{ group: 'men', game_id: 1, date, start_time: null, home: '主', away: '客',
			home_sets_won: null, away_sets_won: null, home_win_prob: null }] };
}

async function slimCalls(utc, readFeed, vars = {}) {
	const calls = [];
	await runMinute(new Date(utc), { ...env, DRY_RUN: false, ...vars }, async (url, options) => {
		if (url === env.FEED_URL) {
			calls.push('feed');
			return readFeed();
		}
		assert.equal(url, endpoint);
		calls.push(JSON.parse(options.body).inputs);
		return new Response(null, { status: 200 });
	});
	return calls;
}

for (const utc of ['07:30', '10:30', '13:15']) {
	test(`賽事日 UTC ${utc} 啟動精簡版`, async () => {
		assert.deepEqual(await slimCalls(`2026-11-20T${utc}:00Z`, () => Response.json(feed('2026-11-20'))),
			['feed', { mode: 'slim' }]);
	});
}

test('只有昨天的賽事時略過，非排程分鐘連賽事檔也不讀', async () => {
	const yesterday = () => Response.json(feed('2026-11-19'));
	assert.deepEqual(await slimCalls('2026-11-20T07:30:00Z', yesterday), ['feed']);
	assert.deepEqual(await slimCalls('2026-11-20T07:31:00Z', yesterday), []);
});

for (const utc of ['2026-11-19T23:59:00Z', '2026-11-20T00:00:00Z']) {
	test(`${utc} 依台北 11 月 20 日判斷`, async () => {
		assert.deepEqual(await slimCalls(utc, () => Response.json(feed('2026-11-20')),
			{ SLIM_TIMES: '07:59,08:00' }), ['feed', { mode: 'slim' }]);
	});
}

for (const [name, read] of [
	['連線失敗', () => { throw new Error('offline'); }],
	['HTTP 503', () => new Response(null, { status: 503 })],
	['無法解析 JSON', () => new Response('<html>')],
	['日期無效', () => Response.json(feed('2026-99-99'))],
	['缺少匯出時間', () => Response.json({ ...feed('2026-11-19'), generated_at: undefined })],
	['賽事格式有誤', () => Response.json({ ...feed('2026-11-20'), matches: [{}] })]
]) {
	test(`賽事檔${name}仍啟動精簡版`, async () => {
		assert.deepEqual(await slimCalls('2026-11-20T13:15:00Z', read), ['feed', { mode: 'slim' }]);
	});
}

test('所有 HTTP 路徑都回 404', async () => {
	for (const method of ['GET', 'POST']) {
		assert.equal((await worker.fetch(new Request('https://test.invalid/score', { method }))).status, 404);
	}
});

test('GitHub 持續回 503 時嘗試三次並讓排程失敗', async () => {
	let attempts = 0;
	await assert.rejects(runMinute(new Date('2026-11-20T01:23:00Z'), env, async () => {
		attempts++;
		return new Response(null, { status: 503 });
	}), /503/);
	assert.equal(attempts, 3);
});

test('GitHub 第二次成功便停止重試', async () => {
	let attempts = 0;
	await runMinute(new Date('2026-11-20T01:23:00Z'), env, async () => {
		return new Response(null, { status: ++attempts === 1 ? 502 : 204 });
	});
	assert.equal(attempts, 2);
});
