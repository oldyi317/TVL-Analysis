"""
TVL 球員名單爬蟲模組
從企業排球聯賽官網抓取球隊球員資料並匯出為 CSV。
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

try:
    from src.utils.logger import get_logger
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    get_logger = logging.getLogger

logger = get_logger(__name__)

# 位置中英對照表（依 CLAUDE.md 第 5 節定義）
POSITION_MAP = {
    "主攻手": "OH",
    "中間手": "MB",
    "副攻手": "OP",
    "舉球員": "S",
    "自由球員": "L",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://tvl.ctvba.org.tw"


def fetch_page(url: str, allow_404: bool = False) -> BeautifulSoup | None:
    """
    抓取網頁並回傳 BeautifulSoup 物件。
    allow_404=True 時，404 回傳 None 且不記錄 error（用於批次探測）。
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if allow_404 and resp.status_code == 404:
            return None
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        logger.error("無法取得頁面 %s: %s", url, e)
        return None


def parse_position(raw: str) -> str | None:
    """從原始字串中比對位置並轉為英文縮寫，例如 '主攻手(隊長)' -> 'OH'。"""
    for zh, en in POSITION_MAP.items():
        if zh in raw:
            return en
    return None


def parse_number(text: str, suffix: str) -> float | None:
    """從含單位字串中提取數值，例如 '187cm' -> 187.0。"""
    m = re.search(rf"([\d.]+)\s*{suffix}", text)
    return float(m.group(1)) if m else None


def parse_player_card(card_div) -> dict:
    """
    解析單一球員卡片（PC_only 區塊），回傳欄位字典。
    任何欄位解析失敗時設為 None，不中斷流程。
    """
    player = {
        "jersey_number": None,
        "name": None,
        "position": None,
        "dob": None,
        "height_cm": None,
        "weight_kg": None,
    }

    try:
        pc_div = card_div.find("div", class_="PC_only")
        if not pc_div:
            logger.warning("找不到 PC_only 區塊，跳過此卡片")
            return player

        # 背號：<h3 class="player_number"><small>#</small>4</h3>
        try:
            num_tag = pc_div.find("h3", class_="player_number")
            if num_tag:
                num_text = num_tag.get_text(strip=True)  # e.g. "#4"
                m = re.search(r"(\d+)", num_text)
                player["jersey_number"] = int(m.group(1)) if m else None
        except Exception as e:
            logger.warning("解析背號失敗: %s", e)

        # 姓名與位置：<h3 class="fs16">李元<span>主攻手(隊長)</span></h3>
        try:
            name_h3 = pc_div.find("h3", class_="fs16")
            if name_h3:
                # 位置在 <span> 內
                span = name_h3.find("span")
                if span:
                    raw_pos = span.get_text(strip=True)
                    player["position"] = parse_position(raw_pos)
                    span.decompose()  # 移除 span 以取得純姓名

                player["name"] = name_h3.get_text(strip=True) or None
        except Exception as e:
            logger.warning("解析姓名/位置失敗: %s", e)

        # 生日、身高、體重：<p> 內的 <i> 標籤
        try:
            p_tag = pc_div.find("p")
            if p_tag:
                italics = p_tag.find_all("i")
                texts = [i.get_text(strip=True) for i in italics]

                # 生日（第一個 <i>），格式 2004.07.17 -> 2004-07-17
                if len(texts) >= 1 and texts[0]:
                    player["dob"] = texts[0].replace(".", "-")

                # 身高（第二個 <i>），格式 187cm -> 187.0
                if len(texts) >= 2:
                    player["height_cm"] = parse_number(texts[1], "cm")

                # 體重（第三個 <i>），格式 80kg -> 80.0
                if len(texts) >= 3:
                    player["weight_kg"] = parse_number(texts[2], "kg")
        except Exception as e:
            logger.warning("解析生日/身高/體重失敗: %s", e)

    except Exception as e:
        logger.error("解析球員卡片時發生未預期錯誤: %s", e)

    return player


def extract_team_name(soup: BeautifulSoup) -> str | None:
    """從頁面 <title> 萃取球隊名稱，並轉換為簡寫。例如 '臺北鯨華女子排球隊 | TVL' -> '臺北鯨華'。"""
    try:
        title_tag = soup.find("title")
        if title_tag:
            raw = title_tag.get_text(strip=True)
            full_name = raw.split("|")[0].strip() or None
            if full_name:
                return TEAM_NAME_SHORT.get(full_name, full_name)
            return None
    except Exception as e:
        logger.warning("萃取球隊名稱失敗: %s", e)
    return None


GENDER_MAP = {"team": "M", "wteam": "F"}

# 官網全名 → 簡寫對應（依 CLAUDE.md 第 7 節）
TEAM_NAME_SHORT = {
    "臺北鯨華女子排球隊": "臺北鯨華",
    "新北中國人纖企業女子排球隊": "新北中纖",
    "台灣電力公司女子排球隊": "高雄台電",
    "義力營造女子排球隊": "義力營造",
    "台灣電力公司男子排球隊": "屏東台電",
    "美津濃男子排球隊": "雲林美津濃",
    "桃園臺產隼鷹排球隊": "桃園臺產",
}


def scrape_team_roster(
    team_id: int,
    prefix: str = "team",
    allow_404: bool = False,
) -> pd.DataFrame:
    """
    抓取指定球隊的球員名單，回傳 DataFrame。
    prefix: 'team'（男子組）或 'wteam'（女子組）。
    若頁面無法取得或無球員資料，回傳空 DataFrame。
    """
    gender = GENDER_MAP.get(prefix, prefix)
    url = f"{BASE_URL}/{prefix}/{team_id}"
    logger.info("正在抓取球隊頁面: %s", url)

    soup = fetch_page(url, allow_404=allow_404)
    if soup is None:
        return pd.DataFrame()

    player_list = soup.find("div", class_="player_list")
    if not player_list:
        logger.info("[%s] %s/%d 無球員列表，跳過", gender, prefix, team_id)
        return pd.DataFrame()

    team_name = extract_team_name(soup)

    cards = player_list.find_all(
        "div", class_="col-md-3", recursive=False
    )
    logger.info(
        "[%s] %s (id=%d) 找到 %d 張球員卡片",
        gender, team_name, team_id, len(cards),
    )

    players = []
    for card in cards:
        player = parse_player_card(card)
        player["team_id"] = team_id
        player["team_name"] = team_name
        player["gender"] = gender
        players.append(player)

    df = pd.DataFrame(players)
    col_order = [
        "gender", "team_id", "team_name", "jersey_number",
        "name", "position", "dob", "height_cm", "weight_kg",
    ]
    return df[[c for c in col_order if c in df.columns]]


def scrape_all_teams(
    prefixes: list[str] = None,
    id_range: range = range(1, 31),
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    批次抓取男女組所有球隊的球員名單。
    prefixes: ['team', 'wteam']，分別對應男子組與女子組。
    對不存在或無球員的頁面靜默跳過。
    每次請求間隔 delay 秒，避免對伺服器造成壓力。
    """
    if prefixes is None:
        prefixes = ["team", "wteam"]

    all_dfs = []
    for prefix in prefixes:
        for team_id in id_range:
            df = scrape_team_roster(
                team_id, prefix=prefix, allow_404=True
            )
            if df.empty:
                continue
            all_dfs.append(df)
            time.sleep(delay)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def main():
    df = scrape_all_teams(
        prefixes=["team", "wteam"],
        id_range=range(1, 31),
        delay=1.0,
    )

    if df.empty:
        logger.error("未取得任何球員資料，程式結束。")
        return

    # 輸出至 CSV
    output_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_teams_roster.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("已儲存至 %s", output_path)

    # 分性別統計摘要
    print("\n===== 爬取完成 =====")
    for gender, label in [("M", "男子組"), ("F", "女子組")]:
        sub = df[df["gender"] == gender]
        if sub.empty:
            print(f"\n【{label}】未抓取到任何球隊")
            continue
        n_teams = sub["team_id"].nunique()
        n_players = len(sub)
        print(f"\n【{label}】球隊數：{n_teams}　球員人數：{n_players}")
        print(sub.groupby(["team_id", "team_name"]).size().to_string())

    print(f"\n總計：{df['team_id'].nunique()} 支球隊，{len(df)} 位球員")
    print(f"\n===== DataFrame 前 5 筆 =====")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
