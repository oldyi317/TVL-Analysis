import { leadersFor, toBoardView, toVolumeRateView } from '#lib/leaders.js';
import { requireSeasonGroup } from '#lib/route.js';
import { seasonGroupEntries } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
	const scope = leadersFor(season.cup_id, group);
	return {
		season,
		group,
		summary: scope ? { through_date: scope.through_date, games: scope.games } : null,
		boards: (scope?.boards ?? []).map((board) => toBoardView(board, group, season.cup_id)),
		volumeRates: (scope?.volume_rates ?? []).map((v) => toVolumeRateView(v, group, season.cup_id))
	};
};
