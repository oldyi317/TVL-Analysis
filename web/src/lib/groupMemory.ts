import { isGroup, type Group } from '#lib/site.js';

export const GROUP_KEY = 'tvl.group';
export const DEFAULT_GROUP: Group = 'men';

export function browserStorage(): Storage | null {
	try {
		return typeof localStorage === 'undefined' ? null : localStorage;
	} catch {
		return null;
	}
}

export function readGroup(storage: Pick<Storage, 'getItem'> | null | undefined): Group {
	try {
		const value = storage?.getItem(GROUP_KEY);
		return value && isGroup(value) ? value : DEFAULT_GROUP;
	} catch {
		return DEFAULT_GROUP;
	}
}

export function writeGroup(storage: Pick<Storage, 'setItem'> | null | undefined, group: Group): void {
	try {
		storage?.setItem(GROUP_KEY, group);
	} catch {
	}
}
