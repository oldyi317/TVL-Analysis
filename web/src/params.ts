import { defineParams } from '@sveltejs/kit/params';

// 建置時由 Node 直接載入，不能匯入專案模組；可接受的值須與 RECORD_KINDS 的 path 一致（records.test.ts 把關）
export const params = defineParams({
	recordKind: (param) => (param === 'cumulative' || param === 'match' ? param : undefined)
});
