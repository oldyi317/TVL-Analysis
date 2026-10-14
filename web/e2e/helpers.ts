import { readFileSync } from 'node:fs';

export const BASE = '/TVL-Analysis';
export const OLD_CUP = 20;
export const OLD_SEASON = '2024-25';
export const readJson = (name: string) =>
	JSON.parse(readFileSync(`src/lib/data/${name}.json`, 'utf8'));
export const seasonsData = readJson('seasons');
export const currentSeason: string = seasonsData.seasons.find(
	(s: { cup_id: number }) => s.cup_id === seasonsData.current_cup_id
).name;
