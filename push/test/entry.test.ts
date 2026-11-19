import { expect, it } from 'vitest';
import * as entry from '../src/index';

// workerd 對入口的其他 export 會在啟動時失敗（Incorrect type for map entry），單元測試本身抓不到
it('入口只 export default', () => {
	expect(Object.keys(entry)).toEqual(['default']);
});
