<script lang="ts">
	import siteJson from '#lib/data/site.json';
	import { formatTaipeiMinute } from '#lib/datetime.js';

	const generatedAt = formatTaipeiMinute(siteJson.generated_at);
</script>

<footer>
	<p>
		資料更新時間 <time datetime={siteJson.generated_at}>{generatedAt}</time>
	</p>
	<p>
		資料來源：TVL 企業排球聯賽官網與其技術統計系統。本站為球迷自製的非官方網站，數據僅供參考，與聯賽及各球隊無關。
	</p>
	<p>
		隊徽與球隊名稱之商標屬各球隊與聯賽所有。程式碼以 MIT 授權公開於
		<a href="https://github.com/oldyi317/TVL-Analysis" target="_blank" rel="noopener">GitHub</a>。
	</p>
	<p>
		<a href="https://github.com/oldyi317/TVL-Analysis/issues" target="_blank" rel="noopener">回報錯誤或許願</a>
		·
		<a href="https://tvl.ctvba.org.tw/" target="_blank" rel="noopener">TVL 官網</a>
	</p>
</footer>

<style>
	footer {
		display: grid;
		gap: var(--space-2);
		padding-block-start: var(--space-4);
		border-block-start: 1px solid var(--line);
		color: var(--text-muted);
		font-size: var(--text-md);
	}

	p {
		margin: 0;
	}

	a {
		color: inherit;
		text-decoration: underline;
	}
</style>
