import assert from 'node:assert/strict';
import test from 'node:test';
import * as entry from '../src/index.ts';

// workerd 對入口的其他 export 會在啟動時失敗（Incorrect type for map entry），單元測試本身抓不到
test('入口只 export default', () => {
	assert.deepEqual(Object.keys(entry), ['default']);
});
