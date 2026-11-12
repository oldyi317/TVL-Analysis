// 不加依賴的靜態伺服器，模擬 GitHub Pages 的子路徑站台。
// CLI：node prototypes/offline/serve.mjs <目錄> [base=/tvl-shell-proto] [port=4180]
// 也可 import：const s = await startServer({ dir, base, port: 0 }); s.url; s.setDir(newDir); await s.close();
import { createServer } from 'node:http';
import { existsSync, statSync, createReadStream } from 'node:fs';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const TYPES = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript; charset=utf-8',
	'.css': 'text/css; charset=utf-8',
	'.json': 'application/json; charset=utf-8',
	'.svg': 'image/svg+xml',
	'.png': 'image/png',
	'.webmanifest': 'application/manifest+json',
	'.txt': 'text/plain; charset=utf-8',
	'.woff2': 'font/woff2'
};

export async function startServer({ dir, base = '/tvl-shell-proto', port = 0, headers = {} }) {
	let root = resolve(dir);
	base = base.replace(/\/$/, '');
	const log = [];

	const isFile = (p) => existsSync(p) && statSync(p).isFile();
	const isDir = (p) => existsSync(p) && statSync(p).isDirectory();

	function send(res, status, file) {
		const type = TYPES[extname(file)] ?? 'application/octet-stream';
		res.writeHead(status, { 'content-type': type, 'cache-control': 'max-age=600', ...headers });
		createReadStream(file).pipe(res);
	}

	function notFound(res) {
		const page = join(root, '404.html');
		if (isFile(page)) return send(res, 404, page);
		res.writeHead(404, { 'content-type': 'text/plain' });
		res.end('Not found');
	}

	const server = createServer((req, res) => {
		const url = new URL(req.url, 'http://x');
		let path;
		try {
			path = decodeURIComponent(url.pathname);
		} catch {
			return notFound(res);
		}
		log.push(`${req.method} ${req.url}`);
		if (path !== base && !path.startsWith(base + '/')) return notFound(res);
		const rel = normalize(path.slice(base.length)).replace(/^[/\\]+/, '');
		const file = resolve(root, rel);
		if (file !== root && !file.startsWith(root + sep)) return notFound(res);

		if (isDir(file)) {
			if (!path.endsWith('/')) {
				res.writeHead(301, { location: `${path}/${url.search}` });
				return res.end();
			}
			const index = join(file, 'index.html');
			return isFile(index) ? send(res, 200, index) : notFound(res);
		}
		if (isFile(file)) return send(res, 200, file);
		if (!extname(file) && isFile(file + '.html')) return send(res, 200, file + '.html');
		return notFound(res);
	});

	await new Promise((ok) => server.listen(port, '127.0.0.1', ok));
	const actual = server.address().port;
	return {
		port: actual,
		base,
		url: `http://127.0.0.1:${actual}${base}/`,
		origin: `http://127.0.0.1:${actual}`,
		log,
		setDir(next) {
			root = resolve(next);
		},
		close() {
			server.closeAllConnections?.();
			return new Promise((ok) => server.close(ok));
		}
	};
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
	const [dir, base, port] = process.argv.slice(2);
	if (!dir) {
		console.error('用法：node serve.mjs <目錄> [base] [port]');
		process.exit(1);
	}
	const s = await startServer({ dir, base, port: Number(port ?? 4180) });
	console.log(`serving ${dir} at ${s.url}`);
}
