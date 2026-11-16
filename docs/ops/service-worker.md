# service worker 維運：換版與自我移除版

第 3 期（spec #231）起站台有一支 service worker（`web/src/service-worker.ts`），負責離線可看與安裝後的啟動。它裝進訪客的瀏覽器之後，網站這邊單純 revert 撤不掉：瀏覽器裡那一份還在，照樣拿存好的檔案回應。要撤得靠本檔後半的自我移除版。

以下命令的網址以正式站 `https://vstat.tw` 為例，預覽站把網址換成 `https://tommyyi0317.github.io/tvl-shell-proto`。

## 換版機制

- **版本怎麼來**：版本字串是 SvelteKit 每次建置產生的時間戳，同一個值寫在 `_app/version.json` 與 `service-worker.js` 裡。每次部署（含每日更新觸發的重新建置）都是一個新版本。預存的 cache 叫 `precache-<版本>`；看過的頁面資料放在 `data`，名稱不帶版本、換版不清。
  線上是哪一版：`curl -s https://vstat.tw/_app/version.json`。
  `service-worker.js` 是不是同一版：`curl -s https://vstat.tw/service-worker.js | grep -c "$(curl -s https://vstat.tw/_app/version.json | grep -o '[0-9]\{6,\}')"`，印出 1 以上就是。
- **什麼時候檢查新版**：只有兩個時機，頁面啟動（整頁載入、註冊完成之後）與回到前景（`visibilitychange` 變成可見）。不定時輪詢。檢查是 `registration.update()`，註冊時設了 `updateViaCache: 'none'`，所以抓 `service-worker.js` 不經瀏覽器的 HTTP 快取。程式在 `web/src/lib/serviceWorker.svelte.ts`。
- **發現新版之後**：瀏覽器在背景把新版的預存整批抓完，新版停在等待狀態，畫面跳出更新條「有新資料，點此更新」。瀏覽到一半不會自動換。
- **什麼時候換上**，三種：
  - 按更新條：新版立刻接手，頁面重新載入、停在同一個網址。同一個瀏覽器裡這個站的其他分頁也會一起重新載入（新版接手時會刪掉舊版的預存，舊分頁留著會缺檔，PR #269）。
  - 不按：這個站的分頁與獨立視窗全部關掉之後，下一次開啟就是新版。
  - 獨立視窗（從主畫面開的）離開超過 30 分鐘再回來：視同重新開啟，有新版就自動換、停在原本那一頁。換上的時間點是新版抓完的那一刻，不一定是回到前景的當下。iPhone 的獨立視窗切到背景不算關掉，所以需要這一條。瀏覽器分頁不管離開多久都只跳更新條。門檻是 `web/src/lib/offline.ts` 的 `AWAY_LIMIT_MS`，剛好 30 分鐘不算。離開的時間只記在記憶體裡。
- **一次換版抓多少**：以一次每日更新的差異估計，壓縮後約 0.78 MB（預算 3 MB）；瀏覽器的 HTTP 快取全空時連沒變的檔也要重抓，共 3.07 MB（PR #269 實測）。獨立視窗另外會把當季的資料檔重新驗證一輪，多半是 304（PR #268）。
- **伺服器端看到的請求**：頁面啟動時 `service-worker.js` 會被要兩次，一次是頁面主動檢查，一次是 Chromium 在頁面載入完之後自己檢查（PR #269 實測，只在 Chromium 量過）。

## 更新最久多久到使用者手上

分兩層看，因為線上看到的內容與離線用的那一份走不同的路。

- **線上開站看到的頁面與資料：部署完成後最多 10 分鐘。** 線上時頁面本身（HTML）與每一頁的資料檔（`__data.json`）都先向網路要，不拿 service worker 存的那一份，所以比分、戰績這些內容不必等換版。延遲來自瀏覽器的 HTTP 快取：GitHub Pages 對所有檔案回 `cache-control: max-age=600`，同一台裝置 10 分鐘內抓過的同一個檔會先用舊的。
  量標頭：`curl -sI https://vstat.tw/ | grep -i cache-control`（2026-11-16 在預覽站對首頁、`service-worker.js`、`_app/version.json`、`shell.html` 量到的都是 `max-age=600`）。
  部署完成的時間：`gh run list --workflow pages.yml --limit 1 --json status,conclusion,updatedAt`。
  兩個例外，都要等換版：
  - 檔名固定的靜態檔（隊徽、`favicon.png` 這些 `web/static` 底下被預存的檔），service worker 先拿預存的那一份，新版換上之前都是舊的。換隊徽之後要等換版才看得到。
  - 網路慢到 3 秒內拿不到頁面時，service worker 改用存好的空殼頁（`web/src/service-worker.ts` 的 `NAVIGATION_TIMEOUT_MS`），這時跑的是舊版程式。
- **離線用的那一份與 service worker 本身：沒有固定上限，取決於使用者什麼時候再開站。** 沒有背景同步，裝置不開站就不會更新。開站之後的順序是：
  1. 下一次啟動或回到前景時發現新版。這一步不受 `max-age=600` 影響（檢查不經 HTTP 快取；`web/e2e/update.e2e.ts` 以回 `max-age=600` 的伺服器測過，iPhone 上的行為由 #242 實機確認）。
  2. 背景抓完新版（抓多少見上一節）。
  3. 照上一節的三種方式換上。所以瀏覽器分頁最晚是「發現新版之後，把這個站的分頁全部關掉再開」；獨立視窗最晚是「發現新版之後，下一次離開超過 30 分鐘再回來」。
- 一直停在前景、不切走也不重新載入的頁面不會檢查新版；站內換頁抓到的資料仍然是新的（同樣最多慢 10 分鐘）。

## 自我移除版

### 它做什麼

建置時設 `VITE_SW_REMOVE=1`，產出的 `service-worker.js` 就是自我移除版，同一次建置的頁面也不再註冊 service worker。部署之後：

- 裝置下一次開站，或已經開著的頁面回到前景時，瀏覽器把它當成新版裝起來。它不等舊版放手就直接接手，接著刪掉這個 origin 底下全部的 cache，再把自己註銷。
- 部署前就開著的舊頁面會自己重新載入一次（跟一般換版一樣），載到的新頁面不再註冊。
- 之後這台裝置開站就是純網站：沒有離線資料，斷網就打不開。主畫面上的圖示還在，點開是沒有離線功能的獨立視窗。
- 沒再開過站的裝置維持原狀，等它下次開站才會清。

清 cache 不分名稱，整個 origin 都清。`vstat.tw` 底下只有這個站；預覽站的 origin 是 `tommyyi0317.github.io`，上面別的專案頁若用了 Cache Storage 也會一起被清。

### 什麼情況下考慮動用

由維護者判斷，沒有固定的判準。可以參考的情況：

- service worker 或它存的檔案出了問題，訪客看到壞掉或過期的畫面，而修正版送不到裝置上（例如新版自己裝不起來）。
- 決定不再提供離線與安裝後的啟動。

多數問題先部署修正版就夠了：線上開站的內容本來就直接向網路要，新版 service worker 在裝置下一次啟動或回到前景時就會被發現。自我移除版是修正版送不到、或來不及等使用者換版時用的。

動用的代價：

- 所有開過站的裝置，離線資料都會被清掉。之後把 service worker 裝回去，每台裝置要從頭預存；獨立視窗連當季資料一共約 3.21 MB（PR #268 量的第一次開啟）。
- **第 4 期上線後動用自我移除版，會讓推播一起停**：推播的收取寫在同一支 service worker 裡，它註銷之後裝置就收不到通知。

### 部署（正式站）

切換要 commit 進 `pages.yml`，不能只手動跑一次：每日更新抓到新資料時會再觸發 `pages.yml`（`scripts/daily_update.sh`），之後任何 commit 進 main 也會。沒有 commit 的話，下一次重新建置就被一般建置蓋回去，service worker 又裝回訪客的瀏覽器。

1. 確認自我移除版還能用。在 `scripts/new-worktree.sh` 開的 worktree 裡（依賴與匯出的資料都備好；沒裝過 Playwright 的瀏覽器就先在 `web/` 跑 `npx playwright install chromium`），從 repo 根目錄跑 `bash scripts/verify-sw-removal.sh`。它建置兩次，再用真的瀏覽器把一般建置換成自我移除版。
   驗證：最後一行是 `✓` 開頭的總結，exit code 0。
2. 從最新的 main 開分支，在 `.github/workflows/pages.yml`「建置靜態站」那一步加上環境變數：

   ```yaml
         - name: 建置靜態站（含 schema 驗證與型別產生）
           working-directory: web
           run: npm run build
           env:
             VITE_SW_REMOVE: '1'
   ```

   驗證：`git diff main --stat` 只有 `.github/workflows/pages.yml` 一個檔；`grep -c "VITE_SW_REMOVE: '1'" .github/workflows/pages.yml` 印出 1。
3. 開 PR、合併進 main。合併後 `pages.yml` 會自己跑。
   驗證：`gh run list --workflow pages.yml --limit 1 --json status,conclusion,headSha,updatedAt`，等到 `status` 是 `completed`、`conclusion` 是 `success`，`headSha` 是合併後的 commit。
4. 確認線上的 `service-worker.js` 已經是自我移除版。
   驗證：`curl -s https://vstat.tw/service-worker.js | grep -c unregister` 印出 1 以上，且 `curl -s https://vstat.tw/service-worker.js | grep -c precache-` 印出 0。
5. 找一台先前開過站的裝置確認。
   - 桌面 Chrome：重新載入一次，等幾秒，在 DevTools 的 Console 貼 `[(await caches.keys()).length, (await navigator.serviceWorker.getRegistrations()).length]`。
     驗證：印出 `[0, 0]`。
   - 手機：開站（或把獨立視窗叫回前景），等幾秒。
     驗證：「更多」面板最下面那一行離線狀態不見了；隔 10 分鐘以上，開飛航模式後重新開站，出現的是瀏覽器或系統的無法連線畫面（10 分鐘內瀏覽器的 HTTP 快取還可能拿得出剛看過的頁面）。
6. 下一次重新建置之後，再確認一次沒有被蓋回去。
   驗證：`gh run list --workflow pages.yml --limit 3 --json createdAt,event,conclusion` 裡有一筆比第 3 步那次新的 run，而且第 4 步的兩條命令結果不變。

部署後 10 分鐘內可能看到註冊又出現：瀏覽器的 HTTP 快取裡還有舊的頁面，舊頁面會再註冊一次。這時註冊到的是自我移除版，它裝好就又把自己註銷，不會留下 cache（`scripts/verify-sw-removal.sh` 有測這條路）。如果再註冊的那個分頁還掛在剛被註銷的註冊上，Chromium 會把註冊留下來；留下的是自我移除版本身，它不存檔、不攔請求，網站照樣是純網站，第 5 步的第二個數字會是 1。

要永久撤掉離線功能，就讓自我移除版一直留在線上：沒再開過站的裝置要等下次開站才清得掉。

### 退回（把 service worker 裝回去）

1. 把上面第 2 步的 commit revert 掉（`git revert <那個 commit>`），開 PR、合併。
   驗證：`grep -c VITE_SW_REMOVE .github/workflows/pages.yml` 印出 0。
2. 等 `pages.yml` 跑完，確認線上換回正式版。
   驗證：`curl -s https://vstat.tw/service-worker.js | grep -c precache-` 印出 1 以上，且 `curl -s https://vstat.tw/service-worker.js | grep -c unregister` 印出 0。
3. 找一台裝置開站確認。
   驗證：「更多」面板最下面先出現「離線資料準備中」，抓完變成「可離線使用，資料更新至 M/D」。

裝回去對每台裝置都是第一次安裝：下次開站才開始預存，在那之前沒有離線資料。第 4 期上線後，通知要不要重新開啟以當時的實測為準。

## 在預覽站演練

第 3 期驗收（#242）照這一節做一次。預覽站是 `https://tommyyi0317.github.io/tvl-shell-proto/`（repo `tommyyi0317/tvl-shell-proto` 的 `gh-pages` 分支放建置產物），建置時 base 要改成 `/tvl-shell-proto`，每個 HTML 要加 noindex。base 的改動不 commit。

前提：預覽站上已經是整合分支的一般建置，至少一台手機開過、「更多」面板顯示「可離線使用」；有 iPhone 的話用主畫面圖示開的獨立視窗。

1. 先在本機確認。從 repo 根目錄跑 `bash scripts/verify-sw-removal.sh`。
   驗證：最後一行是 `✓` 開頭的總結。
2. 把 `web/vite.config.ts` 的 `const base = '';` 改成 `const base = '/tvl-shell-proto';`，在 `web/` 建置自我移除版：`VITE_SW_REMOVE=1 npm run build`。
   驗證：`grep -c unregister build/service-worker.js` 印出 1 以上；`grep -rl service-worker.js build/_app/immutable | wc -l` 印出 0（頁面沒有註冊的程式）；`grep -c /tvl-shell-proto/_app/ build/index.html` 印出 1 以上。
3. 每個 HTML 加 noindex：`grep -rlZ --include='*.html' '<head>' build | xargs -0 sed -i 's|<head>|<head><meta name="robots" content="noindex" />|'`。
   驗證：`grep -rL --include='*.html' noindex build | wc -l` 印出 0。
4. 把 `tommyyi0317/tvl-shell-proto` 的 `gh-pages` 分支清到只剩 `.git`，放進 `web/build/` 的全部內容（含隱藏檔 `.nojekyll`），commit、push，等 GitHub Pages 部署完。
   驗證：`curl -s https://tommyyi0317.github.io/tvl-shell-proto/service-worker.js | grep -c unregister` 印出 1 以上。
5. 在裝置上看。把獨立視窗叫回前景，或在瀏覽器重新載入預覽站，等幾秒。預期看到：
   - 部署前就開著的頁面自己重新載入一次。
   - 「更多」面板最下面那一行離線狀態不見了。
   - 隔 10 分鐘以上，開飛航模式後重新開站，打不開，出現瀏覽器或系統的無法連線畫面。這就是退回純網站。10 分鐘內瀏覽器的 HTTP 快取還可能拿得出剛看過的頁面。
   - 關掉飛航模式，網站照常運作。
   - 桌面 Chrome 另外可以在 Console 貼 `[(await caches.keys()).length, (await navigator.serviceWorker.getRegistrations()).length]`，印出 `[0, 0]`。
6. 收尾，把預覽站部署回一般建置：base 維持 `/tvl-shell-proto`，不帶 `VITE_SW_REMOVE` 重新 `npm run build`，重做第 3、4 步。
   驗證：`curl -s https://tommyyi0317.github.io/tvl-shell-proto/service-worker.js | grep -c precache-` 印出 1 以上；手機重新開站，「更多」面板先出現「離線資料準備中」，再變成「可離線使用，資料更新至 M/D」。
7. 把 `web/vite.config.ts` 改回來。
   驗證：`git status --short web/vite.config.ts` 沒有輸出。
