import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

// 從 @playwright/test 匯入的 test、expect
const directImports = (source: string) =>
	[...source.matchAll(/import\s*\{([^}]*)\}\s*from\s*['"]@playwright\/test['"]/g)]
		.flatMap(([, names]) => names.split(','))
		.map((name) => name.trim().split(/\s+as\s+/)[0])
		.filter((name) => name === 'test' || name === 'expect');

describe('directImports', () => {
	it('抓出從 @playwright/test 以值匯入的 test、expect', () => {
		expect(directImports(`import { expect, test } from '@playwright/test';`)).toEqual(['expect', 'test']);
		expect(directImports(`import { test as base } from "@playwright/test";`)).toEqual(['test']);
		expect(directImports(`import {\n\ttest,\n\ttype Page\n} from '@playwright/test';`)).toEqual(['test']);
	});

	it('型別匯入與 helpers 匯入不算', () => {
		expect(directImports(`import { type Page } from '@playwright/test';`)).toEqual([]);
		expect(directImports(`import { type Locator, type test } from '@playwright/test';`)).toEqual([]);
		expect(directImports(`import type { expect, test } from '@playwright/test';`)).toEqual([]);
		expect(directImports(`import { expect, test } from './helpers.js';`)).toEqual([]);
	});
});

it('e2e spec 一律從 helpers 匯入 test、expect', () => {
	const dir = new URL('.', import.meta.url);
	const offenders = readdirSync(dir, { recursive: true, encoding: 'utf8' })
		.filter((file) => file.endsWith('.e2e.ts'))
		.flatMap((file) => directImports(readFileSync(new URL(file, dir), 'utf8')).map((name) => `${file}：${name}`));
	expect(
		offenders,
		'改從 ./helpers.js 匯入 test、expect（見 e2e/helpers.ts）：那裡的 test 讓 goto／reload 等到 hydration 完成；@playwright/test 只匯入型別'
	).toEqual([]);
});
