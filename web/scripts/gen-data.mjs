// 建置前置：驗證匯出的 JSON 是否符合 schema，並由 schema 產生 TypeScript 型別。
// 任一資料檔缺少或不符 schema → exit 1，讓 build 失敗。
import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import Ajv2020 from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';
import { compile } from 'json-schema-to-typescript';

const DATASETS = ['seasons', 'teams', 'standings'];
const schemaDir = fileURLToPath(new URL('../schema/', import.meta.url));
const dataDir = fileURLToPath(new URL('../src/lib/data/', import.meta.url));

const ajv = new Ajv2020({ allErrors: true });
addFormats(ajv);

let types = '/* 由 scripts/gen-data.mjs 依 web/schema/*.schema.json 產生，請勿手改 */\n\n';
let failed = false;

for (const name of DATASETS) {
	const schema = JSON.parse(await readFile(`${schemaDir}${name}.schema.json`, 'utf8'));
	const dataPath = `${dataDir}${name}.json`;
	if (!existsSync(dataPath)) {
		console.error(`缺少 ${dataPath}，請先從 repo 根目錄執行 python3 -m src.export.site_data`);
		failed = true;
	} else {
		const data = JSON.parse(await readFile(dataPath, 'utf8'));
		const validate = ajv.compile(schema);
		if (!validate(data)) {
			console.error(`${name}.json 不符 schema：`);
			for (const err of validate.errors ?? []) console.error(`  ${err.instancePath} ${err.message}`);
			failed = true;
		}
	}
	types += await compile(schema, schema.title, { bannerComment: '', additionalProperties: false });
}

if (failed) process.exit(1);
await writeFile(`${dataDir}types.d.ts`, types);
console.log(`已驗證 ${DATASETS.join('、')}，型別寫入 ${dataDir}types.d.ts`);
