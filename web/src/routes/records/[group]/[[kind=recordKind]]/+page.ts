import { error } from '@sveltejs/kit';
import { kindFromParam, RECORD_KINDS, recordsFor, toBoardView } from '#lib/records.js';
import { GROUPS, isGroup } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = () =>
	GROUPS.flatMap((group) => RECORD_KINDS.map((k) => (k.path ? { group, kind: k.path } : { group })));

export const load: PageLoad = ({ params }) => {
	const group = params.group;
	const kind = kindFromParam(params.kind);
	if (!isGroup(group) || !kind) error(404, '找不到這個頁面');
	const records = recordsFor(group);
	return {
		group,
		kind,
		coverage: records?.coverage ?? null,
		boards: (records?.boards ?? []).map((board) => toBoardView(board, group))
	};
};
