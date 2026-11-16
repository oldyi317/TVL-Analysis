// 有出處的 User-Agent 字串，逐字抄自來源（2026-11-16 查證）；單元測試與 e2e 共用。
// 內建瀏覽器的字串官方文件沒有寫，取自開源偵測函式庫的測試資料。
export const UA = {
	// iPhone 的 Safari
	// 系統版本凍結在 18_6、Version 是 26.0
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/User-Agent
	iphoneSafari:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1',
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneSafari18:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1',
	// 別的程式用 SFSafariViewController 開的頁，字串與 Safari 相同
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneSafariViewController:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.4 Mobile/15E148 Safari/604.1',

	// iPhone 的其他瀏覽器
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneChrome:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 10_2_1 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) CriOS/58.0.3029.113 Mobile/14D27 Safari/602.1',
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/User-Agent/Firefox
	iphoneFirefox:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 12_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/7.0.4 Mobile/16B91 Safari/605.1.15',
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneEdge:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) EdgiOS/128.0.2739.82 Version/18.0 Mobile/15E148 Safari/604.1',
	// Version 排在 EdgiOS 前面的舊寫法
	// https://github.com/faisalman/ua-parser-js/blob/master/test/data/ua/browser/edge.json
	iphoneEdgeOld:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 11_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.0 EdgiOS/42.1.1.0 Mobile/15F79 Safari/605.1.15',
	// 記號接在 Safari 後面
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneOperaTouch:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1 OPT/5.0.7',

	// iPhone 的內建瀏覽器
	// https://dev.classmethod.jp/articles/liff-vs-line-inappbrowser-user-agent/
	iphoneLine:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Safari Line/15.12.1',
	// https://dev.classmethod.jp/articles/liff-vs-line-inappbrowser-user-agent/
	iphoneLineLiff:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Safari Line/15.12.1 LIFF',
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneFacebook:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/22B83 [FBAN/FBIOS;FBAV/488.0.0.68.101;FBBV/658219612;FBDV/iPhone12,8;FBMD/iPhone;FBSN/iOS;FBSV/18.1;FBSS/2;FBID/phone;FBLC/en_US;FBOP/5;FBRV/0;IABMV/1]',
	// 新版 Facebook，帶 Safari 記號
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneMetaIab:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/22G100 Safari/604.1 MetaIAB Facebook',
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneInstagram:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/22B83 Instagram 354.0.0.29.90 (iPhone12,8; iOS 18_1; en_US; en; scale=2.00; 750x1334; 654111336; IABMV/1)',
	// 新版 Instagram，結尾帶 Safari 記號
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneInstagramSafariToken:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/22F76 Instagram 393.1.0.36.70 (iPhone15,3; iOS 18_5; en_US; en; scale=3.00; 1290x2796; IABMV/1; 776538208) Safari/604.1',
	// Version 與 Safari 記號都有，只多一段結尾
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	iphoneWhatsApp:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0.1 Mobile/15E148 Safari/604.1 [WAiOS/2.25.31]',

	// Android 的 Chrome
	// 型號與系統版本凍結成 Android 10; K
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/User-Agent
	androidChrome:
		'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Mobile Safari/537.36',
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/User-Agent
	androidChromeUnreduced:
		'Mozilla/5.0 (Linux; Android 16; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.12.45 Mobile Safari/537.36',
	// https://www.chromium.org/updates/ua-reduction/
	androidChromePhase4:
		'Mozilla/5.0 (Linux; Android 9; SM-A205U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.0.0 Mobile Safari/537.36',
	// 平板沒有 Mobile 記號
	// https://www.chromium.org/updates/ua-reduction/
	androidChromeTablet:
		'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.0.0 Safari/537.36',

	// Android 的其他瀏覽器
	// https://github.com/faisalman/ua-parser-js/blob/master/test/data/ua/browser/samsung-internet.json
	androidSamsung:
		'Mozilla/5.0 (Linux; Android 6.0.1; SAMSUNG-SM-G925A Build/MMB29K) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/4.0 Chrome/44.0.2403.133 Mobile Safari/537.36',
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/User-Agent
	androidEdge:
		'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36 EdgA/141.0.0.0',
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/User-Agent
	androidOpera:
		'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36 OPR/92.0.0.0',
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/User-Agent/Firefox
	androidFirefox:
		'Mozilla/5.0 (Android 4.4; Mobile; rv:41.0) Gecko/41.0 Firefox/41.0',

	// Android 的內建瀏覽器
	// https://android-developers.googleblog.com/2024/12/user-agent-reduction-on-android-webview.html
	androidWebView:
		'Mozilla/5.0 (Linux; Android 10; K; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/125.000 Mobile Safari/537.36',
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	androidLine:
		'Mozilla/5.0 (Linux; Android 6.0.1; D6653 Build/23.5.A.0.575; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/58.0.3029.83 Mobile Safari/537.36 Line/7.4.0/IAB',
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	androidFacebook:
		'Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/AP2A.240905.003; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/130.0.6723.83 Mobile Safari/537.36 [FB_IAB/FB4A;FBAV/488.0.0.62.79;IABMV/1;]',
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	androidInstagram:
		'Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/AP2A.240905.003; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/130.0.6723.58 Mobile Safari/537.36 Instagram 355.1.0.44.103 Android (34/14; 420dpi; 1080x2205; Google/google; Pixel 8; shiba; shiba; en_US; 658190016)',
	// 沒有 wv 記號，只多一段結尾
	// https://github.com/shalanah/inapp-spy/blob/main/src/tests/mobile.ts
	androidWhatsApp:
		'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.7390.124 Mobile Safari/537.36 [WA4A/2.25.32.75;]',

	// 桌面
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/User-Agent
	desktopChrome:
		'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/User-Agent
	desktopSafari:
		'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15',
	// iPhone 的 Chrome 開「要求電腦版網站」，字串裡沒有 iPhone
	// https://chromium.googlesource.com/chromium/src/+/main/docs/ios/user_agent.md
	iphoneChromeDesktopSite:
		'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/85 Version/11.1.1 Safari/605.1.15'
} as const;
