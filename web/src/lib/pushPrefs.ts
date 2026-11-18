export type PushGroup = 'men' | 'women' | 'both';
export type Prefs = { results: boolean; today: boolean; group: PushGroup };

export const PREFS_KEY = 'tvl.push.prefs';
export const REGISTERED_KEY = 'tvl.push.registeredAt';
export const DEFAULT_PREFS: Prefs = { results: false, today: false, group: 'both' };

const DAY_MS = 24 * 3600_000;

function isPushGroup(value: unknown): value is PushGroup {
	return value === 'men' || value === 'women' || value === 'both';
}

// 偏好與最後登記時間只放這台裝置：讀不到就用預設值，寫不進去不影響這次操作
export function readPrefs(storage: Pick<Storage, 'getItem'> | null | undefined): Prefs {
	try {
		const raw = storage?.getItem(PREFS_KEY);
		const value = raw ? JSON.parse(raw) : null;
		if (typeof value?.results !== 'boolean' || typeof value.today !== 'boolean' || !isPushGroup(value.group)) {
			return { ...DEFAULT_PREFS };
		}
		return { results: value.results, today: value.today, group: value.group };
	} catch {
		return { ...DEFAULT_PREFS };
	}
}

export function writePrefs(storage: Pick<Storage, 'setItem'> | null | undefined, prefs: Prefs): void {
	try {
		storage?.setItem(PREFS_KEY, JSON.stringify(prefs));
	} catch {}
}

export function readRegisteredAt(storage: Pick<Storage, 'getItem'> | null | undefined): number | null {
	try {
		const raw = storage?.getItem(REGISTERED_KEY);
		const value = raw ? Number(raw) : NaN;
		return Number.isFinite(value) ? value : null;
	} catch {
		return null;
	}
}

export function writeRegisteredAt(storage: Pick<Storage, 'setItem'> | null | undefined, at: number): void {
	try {
		storage?.setItem(REGISTERED_KEY, String(at));
	} catch {}
}

// 沒有紀錄、紀錄壞掉或在未來（時鐘被調過）都算到期，寧可多登記一次
export function isDue(registeredAt: number | null, now: number): boolean {
	if (registeredAt === null || registeredAt > now) return true;
	return now - registeredAt >= DAY_MS;
}

export function groupOf(men: boolean, women: boolean): PushGroup {
	if (men && !women) return 'men';
	if (women && !men) return 'women';
	return 'both';
}

export function checksOf(group: PushGroup): { men: boolean; women: boolean } {
	return { men: group !== 'women', women: group !== 'men' };
}

// 每次登記都用裝置上的偏好覆寫，發送服務沒有讀取偏好的入口
export function registerBody(subscription: object, prefs: Prefs) {
	return { ...subscription, results: prefs.results, today: prefs.today, group: prefs.group };
}

export function urlBase64ToBytes(value: string): Uint8Array<ArrayBuffer> {
	const base64 = value.replace(/-/g, '+').replace(/_/g, '/');
	const binary = atob(base64.padEnd(Math.ceil(base64.length / 4) * 4, '='));
	return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}
