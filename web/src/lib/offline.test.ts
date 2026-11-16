import { describe, expect, it } from 'vitest';
import { precachedAssets } from './offline.js';

describe('precachedAssets', () => {
	it('排除四張圖示、推播賽事檔與頁面不會載入的檔，其餘都留', () => {
		const input = [
			'favicon.png',
			'og-image.png',
			'og-image.svg',
			'robots.txt',
			'.nojekyll',
			'icons/apple-touch-icon.png',
			'icons/icon-192.png',
			'icons/icon-512.png',
			'icons/icon-maskable-512.png',
			'push/matches.json',
			'logos/men-1.png',
			'logos/icons-team.png',
			'myicons/a.png',
			'push/other.json',
			'data/push/matches.json'
		];
		expect(precachedAssets(input)).toEqual([
			'favicon.png',
			'logos/men-1.png',
			'logos/icons-team.png',
			'myicons/a.png',
			'push/other.json',
			'data/push/matches.json'
		]);
	});

	it('空清單回空清單', () => {
		expect(precachedAssets([])).toEqual([]);
	});
});
