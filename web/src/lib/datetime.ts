const taipeiParts = new Intl.DateTimeFormat('en-US', {
	timeZone: 'Asia/Taipei',
	year: 'numeric',
	month: '2-digit',
	day: '2-digit',
	hour: '2-digit',
	minute: '2-digit',
	hourCycle: 'h23'
});

function taipeiPartsOf(iso: string): Record<string, string> {
	return Object.fromEntries(taipeiParts.formatToParts(new Date(iso)).map((p) => [p.type, p.value]));
}

export function formatTaipeiMonthDay(iso: string): string {
	const parts = taipeiPartsOf(iso);
	return `${Number(parts.month)}/${Number(parts.day)}`;
}

export function formatTaipeiMinute(iso: string): string {
	const parts = taipeiPartsOf(iso);
	return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`;
}
