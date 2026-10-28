import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import SiteHead from './SiteHead.svelte';
import { SITE_DESCRIPTION } from '#lib/site.js';

function head(title: string, path: string, description?: string) {
	const context = new Map([['__request__', { page: { url: new URL(path, 'http://localhost') } }]]);
	return render(SiteHead, { props: { title, description }, context }).head;
}

describe('共用 head 元件', () => {
	it('輸出 title、description、Open Graph 與 twitter:card，網址皆為正式站絕對網址', () => {
		const html = head('22 男子組｜TVL 數據', '/TVL-Analysis/22/men/');
		expect(html).toContain('<title>22 男子組｜TVL 數據</title>');
		expect(html).toContain(`<meta name="description" content="${SITE_DESCRIPTION}"`);
		expect(html).toContain('<meta property="og:type" content="website"');
		expect(html).toContain('<meta property="og:title" content="22 男子組｜TVL 數據"');
		expect(html).toContain(`<meta property="og:description" content="${SITE_DESCRIPTION}"`);
		expect(html).toContain(
			'<meta property="og:image" content="https://oldyi317.github.io/TVL-Analysis/og-image.png"'
		);
		expect(html).toContain(
			'<meta property="og:url" content="https://oldyi317.github.io/TVL-Analysis/22/men/"'
		);
		expect(html).toContain('<meta name="twitter:card" content="summary_large_image"');
	});

	it('可覆寫 description，根導向頁的 og:url 是站台根', () => {
		const html = head('TVL 數據', '/TVL-Analysis/', '單場描述');
		expect(html).toContain('<meta name="description" content="單場描述"');
		expect(html).toContain('<meta property="og:description" content="單場描述"');
		expect(html).toContain('<meta property="og:url" content="https://oldyi317.github.io/TVL-Analysis/"');
	});
});
