import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import PushPrompt from './PushPrompt.svelte';

it('prerender 時不知道是不是獨立視窗、有沒有問過權限：不輸出提示卡', () => {
	const { body } = render(PushPrompt);
	expect(body).not.toContain('設定通知');
	expect(body).not.toContain('push-prompt');
});
