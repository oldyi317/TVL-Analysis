import { describe, expect, it } from 'vitest';
import { openPush, showPush } from './pushFallback.js';

const SCOPE = 'https://vstat.tw/';
const NAVIGATE = 'https://vstat.tw/2026-27/men/matches/#d-2026-11-21';

type Shown = { title: string; options: NotificationOptions | undefined };

function fakeRegistration() {
	const shown: Shown[] = [];
	return {
		shown,
		scope: SCOPE,
		showNotification: async (title: string, options?: NotificationOptions) => void shown.push({ title, options })
	};
}

const dataOf = (value: unknown) => ({ json: () => value });

describe('showPush', () => {
	it('照發送服務的形狀取標題、內文與要開的網址，顯示成通知', async () => {
		const registration = fakeRegistration();
		await showPush(
			registration,
			dataOf({
				web_push: 8030,
				notification: { title: '男子組賽果', body: '台電 3:1 屏東台電\n雲林美津濃 0:3 臺中太陽神', navigate: NAVIGATE }
			})
		);
		expect(registration.shown).toEqual([
			{
				title: '男子組賽果',
				options: { body: '台電 3:1 屏東台電\n雲林美津濃 0:3 臺中太陽神', data: { navigate: NAVIGATE } }
			}
		]);
	});

	const valid = { title: '今日賽事', body: '14:00 台電 vs 屏東台電', navigate: NAVIGATE };
	const generic = { title: 'VStat 企排', options: { body: '有新消息，點開查看', data: { navigate: SCOPE } } };

	it.each<[string, unknown]>([
		['不是物件', '男子組賽果'],
		['null', null],
		['沒有 notification', { web_push: 8030 }],
		['notification 不是物件', { web_push: 8030, notification: '男子組賽果' }],
		['少了 title', { notification: { ...valid, title: undefined } }],
		['title 是空字串', { notification: { ...valid, title: '' } }],
		['title 不是字串', { notification: { ...valid, title: 3 } }],
		['少了 body', { notification: { ...valid, body: undefined } }],
		['body 是空字串', { notification: { ...valid, body: '' } }],
		['少了 navigate', { notification: { ...valid, navigate: undefined } }],
		['navigate 不是完整網址', { notification: { ...valid, navigate: '/2026-27/men/matches/' } }],
		['navigate 不是 http(s) 網址', { notification: { ...valid, navigate: 'javascript:alert(1)' } }]
	])('內容不認得（%s）：改顯示一則泛用通知，點下去開站台首頁', async (_, payload) => {
		const registration = fakeRegistration();
		await showPush(registration, dataOf(payload));
		expect(registration.shown).toEqual([generic]);
	});

	it('推播沒有內容、或內容不是 JSON：不丟例外，顯示泛用通知', async () => {
		const empty = fakeRegistration();
		await showPush(empty, null);
		expect(empty.shown).toEqual([generic]);

		const broken = fakeRegistration();
		await showPush(broken, {
			json: () => {
				throw new SyntaxError('Unexpected token');
			}
		});
		expect(broken.shown).toEqual([generic]);
	});

	it('不看 web_push 的值，notification 完整就照它顯示', async () => {
		const registration = fakeRegistration();
		await showPush(registration, dataOf({ notification: valid }));
		expect(registration.shown).toEqual([{ title: valid.title, options: { body: valid.body, data: { navigate: NAVIGATE } } }]);
	});

	it('通知不帶 tag，同一天的兩則不會互相取代', async () => {
		const registration = fakeRegistration();
		await showPush(registration, dataOf({ web_push: 8030, notification: valid }));
		expect(registration.shown[0].options).not.toHaveProperty('tag');
	});
});

type Log = string[];

function fakeClient(log: Log, name: string, opts: { failNavigate?: boolean; failFocus?: boolean } = {}) {
	const client = {
		focus: async () => {
			if (opts.failFocus) throw new TypeError('not allowed to focus');
			log.push(`focus ${name}`);
			return client;
		},
		navigate: async (url: string) => {
			if (opts.failNavigate) throw new TypeError('not the active service worker');
			log.push(`navigate ${name} ${url}`);
			return client;
		}
	};
	return client;
}

function fakeClients(log: Log, windows: ReturnType<typeof fakeClient>[]) {
	return {
		matchAll: async (options?: { type?: string }) => {
			log.push(`matchAll ${options?.type}`);
			return windows;
		},
		openWindow: async (url: string) => {
			log.push(`openWindow ${url}`);
			return null;
		}
	};
}

describe('openPush', () => {
	it('沒有這個網站的視窗：開新視窗到通知帶的網址', async () => {
		const log: Log = [];
		await openPush(fakeClients(log, []), { navigate: NAVIGATE }, SCOPE);
		expect(log).toEqual(['matchAll window', `openWindow ${NAVIGATE}`]);
	});

	it('已有這個網站的視窗：聚焦最近用過的那一個並導到該網址，不開新視窗', async () => {
		const log: Log = [];
		const windows = [fakeClient(log, 'a'), fakeClient(log, 'b')];
		await openPush(fakeClients(log, windows), { navigate: NAVIGATE }, SCOPE);
		expect(log).toEqual(['matchAll window', 'focus a', `navigate a ${NAVIGATE}`]);
	});

	it('既有視窗導不過去（不歸這支 service worker 管）：改開新視窗', async () => {
		const log: Log = [];
		await openPush(fakeClients(log, [fakeClient(log, 'a', { failNavigate: true })]), { navigate: NAVIGATE }, SCOPE);
		expect(log).toEqual(['matchAll window', 'focus a', `openWindow ${NAVIGATE}`]);
	});

	it('既有視窗聚焦不了：改開新視窗', async () => {
		const log: Log = [];
		await openPush(fakeClients(log, [fakeClient(log, 'a', { failFocus: true })]), { navigate: NAVIGATE }, SCOPE);
		expect(log).toEqual(['matchAll window', `openWindow ${NAVIGATE}`]);
	});

	it.each<[string, unknown]>([
		['沒有 data', null],
		['data 沒有 navigate', {}],
		['navigate 不是字串', { navigate: 3 }]
	])('通知沒帶網址（%s）：開站台首頁', async (_, data) => {
		const log: Log = [];
		await openPush(fakeClients(log, []), data, SCOPE);
		expect(log).toEqual(['matchAll window', `openWindow ${SCOPE}`]);
	});
});
