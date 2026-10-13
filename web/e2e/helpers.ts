import { readFileSync } from 'node:fs';

export const BASE = '/TVL-Analysis';
export const seasonsData = JSON.parse(readFileSync('src/lib/data/seasons.json', 'utf8'));
export const currentSeason: string = seasonsData.seasons.find(
	(s: { cup_id: number }) => s.cup_id === seasonsData.current_cup_id
).name;
