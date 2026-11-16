import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';

// 自我移除版的真瀏覽器檢查：spec 自己用 e2e/site-server.ts 起伺服器，不起 preview；由 scripts/verify-sw-removal.sh 呼叫
export default defineConfig({ ...base, testMatch: '**/sw-removal.spec.ts', webServer: undefined });
