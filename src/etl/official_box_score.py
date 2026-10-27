import argparse
import json
import re
import sys

import requests
from bs4 import BeautifulSoup

from src.utils.constants import EXT_HEADERS, POSITION_MAP, TVL_BASE_URL


HEADERS = [
    "#", "球員", "位置", "攻擊", "攻擊%", "攔網", "攔網%", "發球", "發球%",
    "接發球", "接發球%", "防守", "防守%", "舉球", "舉球%", "得分",
]
SKILL_COLUMNS = (
    (3, "attack", "points"), (5, "block", "points"),
    (7, "serve", "points"), (9, "receive", "excellent"),
    (11, "dig", "excellent"), (13, "set", "excellent"),
)


def _number(text: str) -> int:
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"無效的非負整數：{text!r}")
    return int(text)


def _player_id(row) -> int | None:
    link = row.find_all("td", recursive=False)[1].find("a", href=re.compile(r"^/w?player/[0-9]+$"))
    return int(link["href"].rsplit("/", 1)[1]) if link else None


def _parse_row(cells: list[str], name: str) -> dict:
    player = {
        "name": re.sub(r"\s*\(C\)\s*$", "", cells[1]).strip(),
        "jersey_number": _number(cells[0]),
        "position": POSITION_MAP.get(cells[2]),
        "position_text": cells[2],
    }
    skill_values = []
    for column, skill, success in SKILL_COLUMNS:
        pair = re.fullmatch(r"([0-9]+)\s*-\s*([0-9]+)", cells[column])
        if pair is None:
            raise ValueError(f"{name} #{cells[0]}：無效的{skill}數值 {cells[column]!r}")
        player[f"{skill}_{success}"] = int(pair[1])
        player[f"{skill}_total"] = int(pair[2])
        skill_values.extend((int(pair[1]), int(pair[2])))
    player["total_points"] = _number(cells[15])
    player["all_zero"] = not any(skill_values)
    return player


def _parse_footer(footer, name: str) -> tuple[int, int, int, int]:
    summary = footer.select_one('td[title="win_points"]')
    totals = footer.find_all("b")
    if summary is None or len(totals) != 3:
        raise ValueError(f"{name}：缺少球員累計、對方失誤或總計")
    return (
        _number(summary.get_text(strip=True)),
        _number(totals[0].get_text(strip=True)),
        _number(totals[1].get_text(strip=True)),
        _number(totals[2].get_text(strip=True)),
    )


def _parse_team(section, name: str, warnings: list, errors: list, gate_errors: list) -> dict:
    if [th.get_text(strip=True) for th in section.select("thead th")] != HEADERS:
        raise ValueError(f"{name}：單場數據欄位順序不符預期")
    body = section.find("tbody")
    footer = section.find("tfoot")
    if body is None or footer is None:
        raise ValueError(f"{name}：缺少球員列或球員累計")
    players = []
    unresolved_rows = []
    rows = body.find_all("tr", recursive=False)
    for row in rows:
        cells = [td.get_text(" ", strip=True) for td in row.find_all("td", recursive=False)]
        if len(cells) != 16:
            raise ValueError(f"{name}：球員列應有 16 欄，實得 {len(cells)}")
        player = _parse_row(cells, name)
        player["official_player_id"] = _player_id(row)
        if player["name"]:
            if player["position"] is None:
                warnings.append(f"{name} #{player['jersey_number']}：未知位置 {cells[2]}")
            players.append(player)
        elif not player["all_zero"] or player["total_points"]:
            errors.append(f"{name} #{player['jersey_number']}：姓名空白但有非零數值")
            unresolved_rows.append(player)
    player_points, display_points, opponent_errors, total_points = _parse_footer(footer, name)
    # 閘門比對頁面自身的完整性：姓名空白的非零列也在頁面累計內，一併加總
    summed_points = sum(p["total_points"] for p in players + unresolved_rows)
    if summed_points != player_points:
        gate_errors.append(
            f"{name}：Σ球員得分 {summed_points} ≠ 球員累計 {player_points}，"
            f"差 {summed_points - player_points:+d}"
        )
    if display_points != player_points:
        warnings.append(f"{name}：頁尾球員累計 {display_points} ≠ 表內球員累計 {player_points}")
    combined = player_points + opponent_errors
    if combined != total_points:
        warnings.append(
            f"{name}：球員累計＋對方失誤 {combined} ≠ 總計 {total_points}，"
            f"差 {combined - total_points:+d}"
        )
    return {
        "name": name, "players": players, "source_row_count": len(rows),
        "unresolved_rows": unresolved_rows,
        "player_points": player_points, "summed_player_points": summed_points,
        "display_player_points": display_points,
        "opponent_errors": opponent_errors, "total_points": total_points,
    }


def parse_box_score(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    teams = []
    errors = []
    gate_errors = []
    warnings = []
    for ident in ("boxscore_1", "boxscore_11"):
        section = soup.find(id=ident)
        heading = section.find_previous("h1") if section else None
        if section is None or heading is None or not heading.get_text(strip=True):
            raise ValueError(f"缺少 {ident} 或球隊名")
        teams.append(_parse_team(section, heading.get_text(strip=True), warnings, errors, gate_errors))
    errors = gate_errors + errors
    return {
        "teams": teams, "warnings": warnings, "errors": errors,
        "gate_errors": gate_errors, "writable": not errors,
    }


def fetch_page(gender: str, game_id: int) -> str:
    """抓官網賽事頁 HTML（男 /game/N、女 /wgame/N）；UA 與逾時沿用 match_crawler。"""
    response = requests.get(page_url(gender, game_id), headers=EXT_HEADERS, timeout=15)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def page_url(gender: str, game_id: int) -> str:
    prefix = "game" if gender == "M" else "wgame"
    return f"{TVL_BASE_URL}/{prefix}/{game_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="官網單場數據 dry-run（不讀寫資料庫）")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--game-id", type=int, required=True)
    parser.add_argument("--gender", choices=("M", "F"), default="M")
    args = parser.parse_args(argv)
    if args.game_id <= 0:
        parser.error("--game-id 必須為正整數")
    url = page_url(args.gender, args.game_id)
    try:
        result = parse_box_score(fetch_page(args.gender, args.game_id))
    except (requests.RequestException, ValueError) as exc:
        print(f"[{args.gender}/{args.game_id}] {url}：{exc}", file=sys.stderr)
        return 1
    report = {
        "url": url, "game_id": args.game_id, "gender": args.gender,
        "gate": "可寫入" if result["writable"] else "不可寫入", **result,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result["writable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
