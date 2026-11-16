import { dev } from '$app/env';
import { resolve } from '$app/paths';

// 範圍與路徑都從 base 推導；dev 沒有建置出的 service worker
export function registerServiceWorker() {
	if (dev || !('serviceWorker' in navigator)) return;
	const scope = resolve('/');
	void navigator.serviceWorker.register(`${scope}service-worker.js`, { scope, type: 'module' });
}
