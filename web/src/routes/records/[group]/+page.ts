import { error } from '@sveltejs/kit';
import { recordsFor, toBoardView } from '#lib/records.js';
import { GROUPS, isGroup } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = () => GROUPS.map((group) => ({ group }));

export const load: PageLoad = ({ params }) => {
	const group = params.group;
	if (!isGroup(group)) error(404, '找不到這個組別');
	const records = recordsFor(group);
	return {
		group,
		coverage: records?.coverage ?? null,
		boards: (records?.boards ?? []).map((board) => toBoardView(board, group))
	};
};
