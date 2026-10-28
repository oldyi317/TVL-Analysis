const taipeiParts = new Intl.DateTimeFormat('en-US', {
	timeZone: 'Asia/Taipei',
	year: 'numeric',
	month: '2-digit',
	day: '2-digit',
	hour: '2-digit',
	minute: '2-digit',
	hourCycle: 'h23'
});

export function formatTaipeiMinute(iso: string): string {
	const parts = Object.fromEntries(
		taipeiParts.formatToParts(new Date(iso)).map((p) => [p.type, p.value])
	);
	return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`;
}
