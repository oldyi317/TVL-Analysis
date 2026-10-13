type Rgb = [number, number, number];

function parseHex(hex: string): Rgb {
	return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as Rgb;
}

function toHex([r, g, b]: Rgb): string {
	return '#' + [r, g, b].map((c) => Math.round(c).toString(16).padStart(2, '0')).join('');
}

// WCAG 2.x 相對亮度
function luminance(hex: string): number {
	const [r, g, b] = parseHex(hex).map((v) => {
		const c = v / 255;
		return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
	});
	return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// WCAG 對比值 1–21；AA 正文門檻 4.5
export function contrastRatio(a: string, b: string): number {
	const la = luminance(a);
	const lb = luminance(b);
	const [hi, lo] = la > lb ? [la, lb] : [lb, la];
	return (hi + 0.05) / (lo + 0.05);
}

// 依 WCAG 相對亮度決定隊標上的文字用黑或白；0.179 這個門檻讓兩者皆至少 4.5:1
export function textColorOn(hex: string): '#000000' | '#ffffff' {
	return luminance(hex) > 0.179 ? '#000000' : '#ffffff';
}

function rgbToHsl([r, g, b]: Rgb): [number, number, number] {
	const [rr, gg, bb] = [r / 255, g / 255, b / 255];
	const max = Math.max(rr, gg, bb);
	const min = Math.min(rr, gg, bb);
	const l = (max + min) / 2;
	if (max === min) return [0, 0, l];
	const d = max - min;
	const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
	let h: number;
	if (max === rr) h = (gg - bb) / d + (gg < bb ? 6 : 0);
	else if (max === gg) h = (bb - rr) / d + 2;
	else h = (rr - gg) / d + 4;
	return [h / 6, s, l];
}

function hslToRgb([h, s, l]: [number, number, number]): Rgb {
	if (s === 0) return [l * 255, l * 255, l * 255];
	const hue = (p: number, q: number, t: number) => {
		if (t < 0) t += 1;
		if (t > 1) t -= 1;
		if (t < 1 / 6) return p + (q - p) * 6 * t;
		if (t < 1 / 2) return q;
		if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
		return p;
	};
	const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
	const p = 2 * l - q;
	return [hue(p, q, h + 1 / 3) * 255, hue(p, q, h) * 255, hue(p, q, h - 1 / 3) * 255];
}

// 把一個顏色（通常是隊色）調到能當 surface 上的文字用：保留色相與飽和度，
// 只往背景的反方向改明度，直到達到 4.5:1。已達標就原樣回傳。
const AA = 4.5;

export function inkOn(hex: string, surface: string): string {
	if (contrastRatio(hex, surface) >= AA) return hex;
	const darken = luminance(surface) > 0.5;
	const [h, s, l0] = rgbToHsl(parseHex(hex));
	let l = l0;
	for (let i = 0; i < 100; i++) {
		l = darken ? l - 0.01 : l + 0.01;
		if (l <= 0 || l >= 1) break;
		const candidate = toHex(hslToRgb([h, s, l]));
		if (contrastRatio(candidate, surface) >= AA) return candidate;
	}
	return darken ? '#000000' : '#ffffff';
}

// 與 src/app.css 的 --surface 同步（tokens.test.ts 守著）；JS 端需要表面色一律從這裡拿
export const SURFACE = { light: '#ffffff', dark: '#1c1f23' } as const;

// 隊色的四個衍生值：填色、填色上的字、淺色模式文字色、深色模式文字色
export function teamPalette(color: string): {
	fill: string;
	onFill: string;
	inkLight: string;
	inkDark: string;
} {
	return {
		fill: color,
		onFill: textColorOn(color),
		inkLight: inkOn(color, SURFACE.light),
		inkDark: inkOn(color, SURFACE.dark)
	};
}
