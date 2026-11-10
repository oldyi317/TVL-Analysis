const hms = (t) => {
	const d = new Date(t);
	const p = (n, w = 2) => String(n).padStart(w, '0');
	return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
};

const report = (id, event, extra = {}) =>
	fetch('/api/receipt', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id, event, at: Date.now(), ...extra }) }).catch(() => {});

// event 自身與原型鏈上（Event.prototype 之前）的屬性名稱
function eventKeys(ev) {
	const keys = new Set();
	for (let o = ev; o && o !== Event.prototype; o = Object.getPrototypeOf(o)) Object.getOwnPropertyNames(o).forEach((k) => keys.add(k));
	return [...keys];
}

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('push', (event) => {
	const receivedAt = Date.now();
	let d = {};
	try {
		d = event.data ? event.data.json() : {};
	} catch (e) {}
	const declarative = d.web_push === 8030;
	const n = (declarative ? d.notification : d) || {};
	const url = declarative ? n.navigate : d.url;
	let id = d.id;
	let sentAt = d.sentAt;
	if (declarative && n.navigate) {
		try {
			const q = new URL(n.navigate).searchParams;
			id = q.get('id');
			sentAt = Number(q.get('sentAt')) || undefined;
		} catch (e) {}
	}
	const diag = { hasEventNotification: 'notification' in event, eventKeys: eventKeys(event), hasData: !!event.data };
	event.waitUntil(
		(async () => {
			// declarative 且系統已帶入通知：交給系統顯示。其餘一律自己顯示（iOS 不允許靜默）。
			const native = declarative && event.notification;
			if (!native) {
				const title = n.title || '推播測試';
				await self.registration.showNotification(declarative ? `〔SW 顯示〕${title}` : title, {
					body: `${n.body || ''}\n發送 ${sentAt ? hms(sentAt) : '?'}／收到 ${hms(receivedAt)}`,
					data: { url, id },
				});
			}
			await report(id || 'unknown', 'push', { ...diag, declarative, native: !!native });
		})(),
	);
});

self.addEventListener('notificationclick', (event) => {
	event.notification.close();
	const { url, id } = event.notification.data || {};
	event.waitUntil(
		(async () => {
			if (id) await report(id, 'click');
			if (url) await self.clients.openWindow(url);
		})(),
	);
});
