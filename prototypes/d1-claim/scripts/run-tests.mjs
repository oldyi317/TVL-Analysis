const [base, token] = process.argv.slice(2);
if (!base || !token) {
	console.error('usage: node scripts/run-tests.mjs <base-url> <token>');
	process.exit(1);
}

const call = async (path, params = {}) => {
	const url = new URL(path, base);
	url.searchParams.set('t', token);
	for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
	const res = await fetch(url);
	return { status: res.status, ...(await res.json()) };
};

const tally = (items) => items.reduce((acc, k) => ({ ...acc, [k]: (acc[k] ?? 0) + 1 }), {});

const out = { base, started_at: new Date().toISOString() };

await call('/init');
await call('/reset');

// 1. 連續呼叫：每次先讀再寫，讀到的值應等於上一次寫入的值
const SEQ_N = 300;
const seq = { n: SEQ_N, stale: [], colos: [], regions: [], primary: [] };
for (let i = 1; i <= SEQ_N; i++) {
	const r = await call('/seq', { i });
	if (r.data.read !== i - 1) seq.stale.push({ i, read: r.data.read });
	seq.colos.push(r.colo);
	seq.regions.push(r.data.read_meta.served_by_region);
	seq.primary.push(String(r.data.read_meta.served_by_primary));
}
out.sequential = {
	n: seq.n,
	stale_count: seq.stale.length,
	stale: seq.stale.slice(0, 10),
	colos: tally(seq.colos),
	served_by_region: tally(seq.regions),
	served_by_primary: tally(seq.primary)
};

// 2. 條件式寫入的回傳值：命中與沒命中
await call('/reset');
out.claim_hit = (await call('/claim', { from: 0 })).data;
out.claim_miss = (await call('/claim', { from: 0 })).data;

// 3. RETURNING
const retHit = await call('/claim-returning', { from: 15 });
const retMiss = await call('/claim-returning', { from: 15 });
out.returning_hit = retHit.data ?? { error: retHit.error };
out.returning_miss = retMiss.data ?? { error: retMiss.error };

// 4. batch：條件式 UPDATE + SELECT
out.batch_hit = (await call('/claim-batch', { from: 30 })).data;
out.batch_miss = (await call('/claim-batch', { from: 30 })).data;

// 5. 唯一鍵：INSERT OR IGNORE、ON CONFLICT DO NOTHING RETURNING、不帶 IGNORE
out.insert_first = (await call('/insert', { key: 'a' })).data;
out.insert_again = (await call('/insert', { key: 'a' })).data;
const insRetFirst = await call('/insert-returning', { key: 'b' });
const insRetAgain = await call('/insert-returning', { key: 'b' });
out.insert_returning_first = insRetFirst.data ?? { error: insRetFirst.error };
out.insert_returning_again = insRetAgain.data ?? { error: insRetAgain.error };
const strict = await call('/insert-strict', { key: 'a' });
out.insert_strict_duplicate = { status: strict.status, error: strict.error, cause: strict.cause };

// 6. 並行搶同一批：每輪 50 個請求帶同一個 from，應只有 1 個改到
const ROUNDS = 20;
const PARALLEL = 50;
await call('/reset');
const race = { rounds: ROUNDS, parallel: PARALLEL, winners_per_round: [], errors: 0, colos: [] };
for (let round = 0; round < ROUNDS; round++) {
	const results = await Promise.all(
		Array.from({ length: PARALLEL }, () => call('/claim', { from: round * 15 }))
	);
	race.errors += results.filter((r) => r.status !== 200).length;
	race.winners_per_round.push(results.filter((r) => r.data?.changes === 1).length);
	race.colos.push(...results.map((r) => r.colo));
}
const finalProgress = (await call('/progress')).data.results[0].next_offset;
out.race_claim = {
	rounds: race.rounds,
	parallel: race.parallel,
	winners_per_round: tally(race.winners_per_round),
	errors: race.errors,
	colos: tally(race.colos),
	final_offset: finalProgress,
	expected_offset: ROUNDS * 15
};

// 7. 並行寫同一個唯一鍵：每輪 50 個請求，應只有 1 個寫入
const raceInsert = { winners_per_round: [], errors: 0 };
for (let round = 0; round < ROUNDS; round++) {
	const results = await Promise.all(
		Array.from({ length: PARALLEL }, () => call('/insert', { key: `race-${round}` }))
	);
	raceInsert.errors += results.filter((r) => r.status !== 200).length;
	raceInsert.winners_per_round.push(results.filter((r) => r.data?.changes === 1).length);
}
out.race_insert = {
	rounds: ROUNDS,
	parallel: PARALLEL,
	winners_per_round: tally(raceInsert.winners_per_round),
	errors: raceInsert.errors
};

out.finished_at = new Date().toISOString();
console.log(JSON.stringify(out, null, 2));
