import { afterNavigate } from '$app/navigation';

// 原型專用：殼層變體與站內歷史深度
export type Variant = 'A' | 'B' | 'C' | 'D';
export const LABELS: Record<Variant, string> = { A: 'A 現狀', B: 'B 吸頂 pill 列', C: 'C 底部分頁・單一按鈕', D: 'D 底部分頁・組別與賽季分開' };
export const ORDER: Variant[] = ['A', 'B', 'C', 'D'];

export const shell = $state<{ variant: Variant; depth: number }>({ variant: 'A', depth: 0 });

export function syncVariant() {
	const v = document.documentElement.dataset.shell;
	shell.variant = v === 'B' || v === 'C' || v === 'D' ? v : 'A';
}

export function setVariant(v: Variant) {
	shell.variant = v;
	document.documentElement.dataset.shell = v;
	try {
		localStorage.setItem('proto-shell-variant', v);
	} catch {}
}

// 在 layout 初始化時呼叫一次
export function trackNav() {
	afterNavigate((n) => {
		if (n.type === 'enter') shell.depth = 0;
		else if (n.type === 'popstate') shell.depth = Math.max(0, shell.depth + (n.delta ?? 0));
		else shell.depth += 1;
	});
}
