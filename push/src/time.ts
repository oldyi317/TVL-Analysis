export interface TaipeiTime {
	date: string;
	minutes: number;
}

export interface Window {
	start: number;
	end: number;
}

const TAIPEI_OFFSET_MS = 8 * 60 * 60 * 1000;
const DAY_MS = 24 * 60 * 60 * 1000;

// 台北固定 UTC+8、沒有日光節約：平移八小時後讀 UTC 欄位。
export function taipeiTime(now: Date): TaipeiTime {
	const shifted = new Date(now.getTime() + TAIPEI_OFFSET_MS);
	return {
		date: shifted.toISOString().slice(0, 10),
		minutes: shifted.getUTCHours() * 60 + shifted.getUTCMinutes()
	};
}

export function daysBetween(from: string, to: string): number {
	return Math.round((Date.parse(to) - Date.parse(from)) / DAY_MS);
}

export function parseWindow(text: string): Window | null {
	const match = /^(\d{2}):(\d{2})-(\d{2}):(\d{2})$/.exec(text);
	if (!match) return null;
	const [startHour, startMinute, endHour, endMinute] = match.slice(1).map(Number);
	if (startHour > 23 || endHour > 23 || startMinute > 59 || endMinute > 59) return null;
	const start = startHour * 60 + startMinute;
	const end = endHour * 60 + endMinute;
	if (start > end) return null;
	return { start, end };
}

// 閉區間：結束那一分鐘也算在內。
export function inWindow(window: Window, minutes: number): boolean {
	return minutes >= window.start && minutes <= window.end;
}
