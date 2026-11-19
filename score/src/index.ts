import { runMinute, type Env } from './schedule.ts';

export default {
	scheduled(event, env) {
		return runMinute(new Date(event.scheduledTime), env);
	},
	fetch() {
		return new Response(null, { status: 404 });
	}
} satisfies ExportedHandler<Env>;
