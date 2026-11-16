export type InstallGuide = 'ios-safari' | 'ios-other' | 'android-chrome' | 'android-other' | 'standalone' | 'desktop';

const SAFARI_SHAPE = /\(KHTML, like Gecko\) Version\/\S+ Mobile\/\S+ Safari\/\S+$/;
const CHROME_SHAPE = /\(KHTML, like Gecko\) Chrome\/\S+ (?:Mobile )?Safari\/\S+$/;

// 比對整串「純 Safari／Chrome 的形狀」，不列排除清單：其他瀏覽器與 LINE、Facebook 這類內建瀏覽器
// 都在這兩種 UA 裡插入或附加自己的記號，清單一定漏，只認得純的就不必認識每一家。
// 不看 UA 裡的系統版本號：新版 iOS 把它凍結了（fixture 裡 Version/26.0 的 Safari 寫的是 iPhone OS 18_6）。
export function installGuideOf(env: { userAgent: string; standalone: boolean }): InstallGuide {
	if (env.standalone) return 'standalone';
	const ua = env.userAgent;
	if (ua.includes('iPhone')) return SAFARI_SHAPE.test(ua) ? 'ios-safari' : 'ios-other';
	if (ua.includes('Android')) return CHROME_SHAPE.test(ua) ? 'android-chrome' : 'android-other';
	return 'desktop';
}

export function inPhoneBrowserTab(guide: InstallGuide): boolean {
	return guide !== 'standalone' && guide !== 'desktop';
}
