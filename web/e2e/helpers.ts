import { readFileSync } from 'node:fs';
import { KINDS, pickSeason, type Kind } from './pick-season.js';

export const BASE = '/TVL-Analysis';
export const OLD_CUP = 20;
export const OLD_SEASON = '2024-25';
export const readJson = (name: string) =>
	JSON.parse(readFileSync(`src/lib/data/${name}.json`, 'utf8'));
export const seasonsData = readJson('seasons');
export const currentSeason: string = seasonsData.seasons.find(
	(s: { cup_id: number }) => s.cup_id === seasonsData.current_cup_id
).name;

// 讀匯出 JSON 的薄包裝：給 spec 取「最近一個有該資料的賽季」
export const findSeason = (kind: Kind, group = 'men') => {
	const { file } = KINDS[kind];
	return pickSeason(kind, group, {
		rows: readJson(file)[file],
		seasons: seasonsData.seasons,
		currentCupId: seasonsData.current_cup_id
	});
};
