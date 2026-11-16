import { installGuideOf, type InstallGuide } from '#lib/installGuide.js';
import { isStandalone } from '#lib/serviceWorker.svelte.js';

// TypeScript 的 DOM 型別沒有這個事件；只有 Chromium 系瀏覽器會送，iOS 不會
interface BeforeInstallPromptEvent extends Event {
	prompt(): Promise<void>;
	userChoice: Promise<unknown>;
}

// guide 在瀏覽器裡判斷前是 null（prerender 的 HTML 不知道瀏覽器種類）
export const installPrompt = $state<{ guide: InstallGuide | null; promptReady: boolean }>({
	guide: null,
	promptReady: false
});

// 事件只能用一次，放一般變數，不進 $state
let deferred: BeforeInstallPromptEvent | undefined;

// 只在 Android Chrome 掛監聽：其他瀏覽器的頁面上沒有安裝鈕，不動瀏覽器自己的行為
export function startInstall() {
	installPrompt.guide = installGuideOf({ userAgent: navigator.userAgent, standalone: isStandalone() });
	if (installPrompt.guide !== 'android-chrome') return;
	window.addEventListener('beforeinstallprompt', (event) => {
		// 擋掉瀏覽器自己跳出的安裝小橫條，留著事件等使用者按鈕
		event.preventDefault();
		deferred = event as BeforeInstallPromptEvent;
		installPrompt.promptReady = true;
	});
	window.addEventListener('appinstalled', () => {
		deferred = undefined;
		installPrompt.promptReady = false;
	});
}

// prompt() 必須在點擊處理函式裡直接呼叫，前面不能有 await
export async function promptInstall() {
	const event = deferred;
	if (!event) return;
	// 先取走：對話框還開著時再點一次，就沒有事件可用
	deferred = undefined;
	try {
		await event.prompt();
		await event.userChoice;
	} finally {
		// 使用者取消後瀏覽器通常會再送新的事件，可能已經到了
		installPrompt.promptReady = deferred !== undefined;
	}
}
