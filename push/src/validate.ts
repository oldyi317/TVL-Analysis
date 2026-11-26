import { fromBase64Url } from './webpush';

const GROUPS = ['men', 'women', 'both'] as const;

type GroupPref = (typeof GROUPS)[number];

export interface Subscription {
	endpoint: string;
	p256dh: string;
	auth: string;
	wantResults: boolean;
	wantToday: boolean;
	group: GroupPref;
}

type Parsed<T> = { ok: true; value: T } | { ok: false; error: 'invalid_body' | 'endpoint_not_allowed' };

const MAX_ENDPOINT_LENGTH = 2048;

// 依據見 push/README.md「推播服務網域」。
const EXACT_HOSTS = new Set(['fcm.googleapis.com', 'updates.push.services.mozilla.com']);
const HOST_SUFFIXES = ['.push.apple.com', '.notify.windows.com'];

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isEndpointString(value: unknown): value is string {
	return typeof value === 'string' && value.length > 0 && value.length <= MAX_ENDPOINT_LENGTH;
}

function isGroupPref(value: unknown): value is GroupPref {
	return GROUPS.some((group) => group === value);
}

// 瀏覽器給的是無填充的 base64url；有人補了 '=' 也收，存成無填充。p256dh 解開是 65 bytes，auth 是 16 bytes。
function parseKey(value: unknown, length: number): string | null {
	if (typeof value !== 'string') return null;
	const key = value.replace(/=+$/, '');
	return key.length === length && /^[A-Za-z0-9_-]+$/.test(key) ? key : null;
}

// 長度與字元都對，也可能不是 P-256 曲線上的點：發送時匯入會失敗，那一筆送不出去也不會被刪，一直佔著上限。
// 所以登記時就照發送時的參數匯入一次，匯不進去的不收。
async function isP256PublicKey(p256dh: string): Promise<boolean> {
	try {
		await crypto.subtle.importKey('raw', fromBase64Url(p256dh), { name: 'ECDH', namedCurve: 'P-256' }, false, []);
		return true;
	} catch {
		return false;
	}
}

// 同一個推播位址的不同寫法（主機大小寫、:443、#片段）正規化成同一個字串，才不會各佔一筆。
function normalizeEndpoint(endpoint: string): URL | null {
	try {
		const url = new URL(endpoint);
		url.hash = '';
		return url;
	} catch {
		return null;
	}
}

function isAllowedEndpoint(url: URL): boolean {
	if (url.protocol !== 'https:' || url.username || url.password || url.port) return false;
	return EXACT_HOSTS.has(url.hostname) || HOST_SUFFIXES.some((suffix) => url.hostname.endsWith(suffix));
}

export async function parseSubscribe(body: unknown): Promise<Parsed<Subscription>> {
	if (!isRecord(body) || !isRecord(body.keys)) return { ok: false, error: 'invalid_body' };
	const { endpoint, results, today, group } = body;
	const p256dh = parseKey(body.keys.p256dh, 87);
	const auth = parseKey(body.keys.auth, 22);
	if (!isEndpointString(endpoint) || !p256dh || !auth) return { ok: false, error: 'invalid_body' };
	if (typeof results !== 'boolean' || typeof today !== 'boolean') return { ok: false, error: 'invalid_body' };
	if (!results && !today) return { ok: false, error: 'invalid_body' };
	if (!isGroupPref(group)) return { ok: false, error: 'invalid_body' };
	const url = normalizeEndpoint(endpoint);
	if (!url || !isAllowedEndpoint(url)) return { ok: false, error: 'endpoint_not_allowed' };
	if (!(await isP256PublicKey(p256dh))) return { ok: false, error: 'invalid_body' };
	return { ok: true, value: { endpoint: url.href, p256dh, auth, wantResults: results, wantToday: today, group } };
}

export function parseUnsubscribe(body: unknown): Parsed<string> {
	if (!isRecord(body) || !isEndpointString(body.endpoint)) return { ok: false, error: 'invalid_body' };
	return { ok: true, value: normalizeEndpoint(body.endpoint)?.href ?? body.endpoint };
}
