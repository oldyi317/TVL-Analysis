// 依 WCAG 相對亮度決定隊標上的文字用黑或白
export function textColorOn(hex: string): '#000000' | '#ffffff' {
	const channels = [1, 3, 5].map((i) => {
		const c = parseInt(hex.slice(i, i + 2), 16) / 255;
		return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
	});
	const luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
	return luminance > 0.179 ? '#000000' : '#ffffff';
}
