import { readFileSync } from 'node:fs';
import { createServer, type IncomingHttpHeaders } from 'node:http';
import type { AddressInfo } from 'node:net';
import { extname, join, normalize } from 'node:path';

const TYPES: Record<string, string> = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript; charset=utf-8',
	'.css': 'text/css; charset=utf-8',
	'.json': 'application/json; charset=utf-8',
	'.png': 'image/png',
	'.svg': 'image/svg+xml',
	'.webmanifest': 'application/manifest+json'
};

const BUILD = 'build';
const SCRIPT = '/service-worker.js';

// 供應 build/ 的測試用靜態伺服器：瀏覽器抓 service-worker.js 主腳本的請求 Playwright 的路由碰不到，
// 所以「部署新版」只能由伺服器換腳本；所有回應帶 max-age=600，與 GitHub Pages 一致
export async function startSite(base: string, version: string) {
	const original = readFileSync(join(BUILD, SCRIPT), 'utf8');
	const next = `${version}-next`;
	const upgraded = original.replaceAll(version, next);
	let deployed = false;
	let root = BUILD;
	const scriptRequests: IncomingHttpHeaders[] = [];
	const holds = new Map<string, { gate: Promise<void>; arrive: () => void }>();

	const server = createServer(async (req, res) => {
		const url = new URL(req.url ?? '/', 'http://localhost');
		const path = url.pathname.startsWith(base) ? url.pathname.slice(base.length) || '/' : url.pathname;
		const send = (status: number, type: string, body: string | Buffer) => {
			res.writeHead(status, { 'content-type': type, 'cache-control': 'max-age=600' });
			res.end(body);
		};
		for (const [suffix, hold] of holds) {
			if (!path.endsWith(suffix)) continue;
			hold.arrive();
			await hold.gate;
		}
		if (path === SCRIPT) {
			scriptRequests.push(req.headers);
			if (root !== BUILD) return send(200, TYPES['.js'], readFileSync(join(root, SCRIPT)));
			return send(200, TYPES['.js'], deployed ? upgraded : original);
		}
		const file = normalize(join(root, path.endsWith('/') ? `${path}index.html` : path));
		if (!file.startsWith(`${root}/`)) return send(404, 'text/plain', 'not found');
		try {
			send(200, TYPES[extname(file)] ?? 'application/octet-stream', readFileSync(file));
		} catch {
			send(404, 'text/plain', 'not found');
		}
	});
	await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
	const origin = `http://localhost:${(server.address() as AddressInfo).port}`;

	return {
		origin,
		next,
		scriptRequests,
		// 給目錄就改從那個目錄供應所有請求（含 service-worker.js），例如自我移除版的建置
		deploy(dir?: string) {
			deployed = true;
			if (dir) root = dir;
		},
		// 路徑以 suffix 結尾的請求會被扣住，直到 release
		hold(suffix: string) {
			let arrive!: () => void;
			let release!: () => void;
			const arrived = new Promise<void>((resolve) => (arrive = resolve));
			const gate = new Promise<void>((resolve) => (release = resolve));
			holds.set(suffix, { gate, arrive });
			return {
				arrived,
				release: () => {
					holds.delete(suffix);
					release();
				}
			};
		},
		async close() {
			for (const hold of holds.values()) hold.arrive();
			server.closeAllConnections();
			await new Promise<void>((resolve) => server.close(() => resolve()));
		}
	};
}

export type Site = Awaited<ReturnType<typeof startSite>>;
