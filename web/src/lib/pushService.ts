// 發送服務的網址與 VAPID 公鑰是 build 時的設定；預設是正式環境，預覽站 build 時用 VITE_PUSH_URL、VITE_PUSH_PUBLIC_KEY 換掉
export const PUSH_URL: string = import.meta.env.VITE_PUSH_URL || 'https://push.vstat.tw';
export const PUSH_PUBLIC_KEY: string =
	import.meta.env.VITE_PUSH_PUBLIC_KEY ||
	'BGDlCqN1KpK7X7zAF4M5gsbj4zJ8lzHYDut1DEqohJOpaAUqlDZJyshcc5x2-p_CEG03u5EjoBDZfUeaqI0kIR0';
