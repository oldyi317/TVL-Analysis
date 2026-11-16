import { deleteSubscription, upsertSubscription } from './store';
import { parseSubscribe, parseUnsubscribe } from './validate';

export interface Env {
	DB: D1Database;
	ALLOWED_ORIGIN: string;
}

const MAX_BODY_BYTES = 4096;

type Handler = (body: unknown, env: Env) => Promise<Response>;

const ROUTES: Record<string, Handler> = {
	'/subscribe': subscribe,
	'/unsubscribe': unsubscribe
};

function json(status: number, data: unknown, headers: Record<string, string> = {}): Response {
	return new Response(JSON.stringify(data), {
		status,
		headers: { 'Content-Type': 'application/json; charset=utf-8', ...headers }
	});
}

function corsHeaders(origin: string): Record<string, string> {
	return { 'Access-Control-Allow-Origin': origin, Vary: 'Origin' };
}

function preflight(origin: string): Response {
	return new Response(null, {
		status: 204,
		headers: {
			...corsHeaders(origin),
			'Access-Control-Allow-Methods': 'POST',
			'Access-Control-Allow-Headers': 'Content-Type',
			'Access-Control-Max-Age': '86400'
		}
	});
}

async function readJson(request: Request): Promise<{ body: unknown } | { status: number; error: string }> {
	const declared = Number(request.headers.get('Content-Length') ?? 0);
	if (declared > MAX_BODY_BYTES) return { status: 413, error: 'body_too_large' };
	const text = await request.text();
	if (new TextEncoder().encode(text).length > MAX_BODY_BYTES) return { status: 413, error: 'body_too_large' };
	try {
		return { body: JSON.parse(text) };
	} catch {
		return { status: 400, error: 'invalid_body' };
	}
}

async function subscribe(body: unknown, env: Env): Promise<Response> {
	const parsed = parseSubscribe(body);
	if (!parsed.ok) return json(400, { error: parsed.error });
	const saved = await upsertSubscription(env.DB, parsed.value, new Date());
	return saved ? json(200, { ok: true }) : json(503, { error: 'subscription_limit_reached' });
}

async function unsubscribe(body: unknown, env: Env): Promise<Response> {
	const parsed = parseUnsubscribe(body);
	if (!parsed.ok) return json(400, { error: parsed.error });
	await deleteSubscription(env.DB, parsed.value);
	return json(200, { ok: true });
}

async function route(request: Request, env: Env, origin: string): Promise<Response> {
	const handler = ROUTES[new URL(request.url).pathname];
	if (!handler) return json(404, { error: 'not_found' });
	if (request.method === 'OPTIONS') return preflight(origin);
	if (request.method !== 'POST') return json(405, { error: 'method_not_allowed' }, { Allow: 'POST, OPTIONS' });
	const read = await readJson(request);
	if ('error' in read) return json(read.status, { error: read.error });
	return handler(read.body, env);
}

export default {
	async fetch(request, env): Promise<Response> {
		const origin = request.headers.get('Origin');
		if (!origin || origin !== env.ALLOWED_ORIGIN) return json(403, { error: 'origin_not_allowed' });
		const response = await route(request, env, origin);
		for (const [name, value] of Object.entries(corsHeaders(origin))) response.headers.set(name, value);
		return response;
	}
} satisfies ExportedHandler<Env>;
