import { Validator, type Schema } from '@cfworker/json-schema';
import schema from '../../web/schema/push_matches.schema.json' with { type: 'json' };
import { taipeiTime } from '../../push/src/time.ts';

const feedSchema = new Validator(schema as Schema, '2020-12');

export interface Env {
	FULL_TIME: string;
	SLIM_TIMES: string;
	FEED_URL: string;
	GITHUB_REPO: string;
	GITHUB_WORKFLOW: string;
	DRY_RUN: boolean;
	GITHUB_TOKEN: string;
}

async function hasMatches(date: string, url: string, request: typeof fetch): Promise<boolean> {
	try {
		const response = await request(url);
		if (!response.ok) return true;
		const feed = await response.json<{ matches: { date: string }[] }>();
		return !feedSchema.validate(feed).valid || feed.matches.some(match => match.date === date);
	} catch {
		return true;
	}
}

export async function runMinute(now: Date, env: Env, request = fetch): Promise<void> {
	const { date, minutes } = taipeiTime(now);
	const hhmm = [Math.floor(minutes / 60), minutes % 60].map(n => String(n).padStart(2, '0')).join(':');
	const mode = hhmm === env.FULL_TIME ? 'full' : 'slim';
	if (mode === 'slim') {
		if (!env.SLIM_TIMES.split(',').includes(hhmm)) return;
		if (!await hasMatches(date, env.FEED_URL, request)) return;
	}

	const url = `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/${env.GITHUB_WORKFLOW}/dispatches`;
	const options = {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${env.GITHUB_TOKEN}`,
			Accept: 'application/vnd.github+json',
			'Content-Type': 'application/json',
			'User-Agent': 'tvl-score',
			'X-GitHub-Api-Version': '2026-03-10'
		},
		body: JSON.stringify({ ref: 'main', inputs: { mode, ...(env.DRY_RUN && { dry_run: true }) } })
	};
	for (let attempt = 1; attempt <= 3; attempt++) {
		const response = await request(url, options);
		if (response.ok) return;
		if (attempt === 3) throw new Error(`GitHub workflow dispatch 回應 ${response.status}`);
	}
}
