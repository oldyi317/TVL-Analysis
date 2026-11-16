import { describe, expect, it } from 'vitest';
import { inPhoneBrowserTab, installGuideOf, type InstallGuide } from '#lib/installGuide.js';
import { UA } from '#lib/installGuide.fixtures.js';

type Case = [keyof typeof UA, InstallGuide];

const tab = (key: keyof typeof UA) => installGuideOf({ userAgent: UA[key], standalone: false });

describe('installGuideOf', () => {
	it.each<Case>([
		['iphoneSafari', 'ios-safari'],
		['iphoneSafari18', 'ios-safari'],
		// 已知限制：別的程式用 SFSafariViewController 開的頁，字串與 Safari 完全相同，分不出來
		['iphoneSafariViewController', 'ios-safari'],
		['iphoneChrome', 'ios-other'],
		['iphoneFirefox', 'ios-other'],
		['iphoneEdge', 'ios-other'],
		['iphoneEdgeOld', 'ios-other'],
		['iphoneOperaTouch', 'ios-other'],
		['iphoneLine', 'ios-other'],
		['iphoneLineLiff', 'ios-other'],
		['iphoneFacebook', 'ios-other'],
		['iphoneMetaIab', 'ios-other'],
		['iphoneInstagram', 'ios-other'],
		['iphoneInstagramSafariToken', 'ios-other'],
		['iphoneWhatsApp', 'ios-other'],
		['androidChrome', 'android-chrome'],
		['androidChromeUnreduced', 'android-chrome'],
		['androidChromePhase4', 'android-chrome'],
		['androidChromeTablet', 'android-chrome'],
		['androidSamsung', 'android-other'],
		['androidEdge', 'android-other'],
		['androidOpera', 'android-other'],
		['androidFirefox', 'android-other'],
		['androidWebView', 'android-other'],
		['androidLine', 'android-other'],
		['androidFacebook', 'android-other'],
		['androidInstagram', 'android-other'],
		['androidWhatsApp', 'android-other'],
		['desktopChrome', 'desktop'],
		['desktopSafari', 'desktop'],
		['iphoneChromeDesktopSite', 'desktop']
	])('%s 是 %s', (key, expected) => {
		expect(tab(key)).toBe(expected);
	});

	it.each<keyof typeof UA>(['iphoneSafari', 'androidChrome', 'desktopChrome'])('獨立視窗不看 UA：%s 回 standalone', (key) => {
		expect(installGuideOf({ userAgent: UA[key], standalone: true })).toBe('standalone');
	});
});

describe('inPhoneBrowserTab', () => {
	it.each<[InstallGuide, boolean]>([
		['ios-safari', true],
		['ios-other', true],
		['android-chrome', true],
		['android-other', true],
		['standalone', false],
		['desktop', false]
	])('%s 是 %s', (guide, expected) => {
		expect(inPhoneBrowserTab(guide)).toBe(expected);
	});
});
