"""
TVL 資料清洗模組
負責 raw data 進入資料庫前的品質把關：型別轉換、格式驗證、重複偵測、異常值標記。
遵守零臆測原則：缺失值保留 None，絕不插補。
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "all_teams_roster.csv"
CLEANED_CSV = PROJECT_ROOT / "data" / "processed" / "all_teams_roster_cleaned.csv"

# 合法位置代碼（依 CLAUDE.md 第 5 節）
VALID_POSITIONS = {"OH", "MB", "OP", "S", "L"}
VALID_GENDERS = {"M", "F"}

# 身高體重合理範圍（用於異常值警告，不會刪除或修改資料）
HEIGHT_RANGE = (140.0, 220.0)
WEIGHT_RANGE = (40.0, 150.0)


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """讀取原始 CSV。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("讀取原始資料：%d 筆", len(df))
    return df


# ── 型別轉換 ─────────────────────────────────────────────────────

def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    強制數值欄位轉型：去除殘留單位字串後轉為 Float。
    字串無法轉換者設為 NaN（後續統一轉 None）。
    """
    for col, suffix in [("height_cm", "cm"), ("weight_kg", "kg")]:
        if col not in df.columns:
            continue
        # 若欄位混入字串（如 "187cm"），先清除單位
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(rf"\s*{suffix}\s*", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # jersey_number 轉為可空整數
    if "jersey_number" in df.columns:
        df["jersey_number"] = pd.to_numeric(
            df["jersey_number"], errors="coerce"
        ).astype("Int64")

    return df


# ── 日期格式驗證 ──────────────────────────────────────────────────

def validate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """確認 dob 欄位符合 YYYY-MM-DD 格式，不合格者設為 None。"""
    if "dob" not in df.columns:
        return df

    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def check_date(val):
        if pd.isna(val) or val is None:
            return None
        val = str(val).strip()
        # 嘗試修正常見格式：YYYY.MM.DD → YYYY-MM-DD
        val = val.replace(".", "-").replace("/", "-")
        if date_pattern.match(val):
            return val
        logger.warning("日期格式異常，設為 None：%s", val)
        return None

    df["dob"] = df["dob"].apply(check_date)
    return df


# ── 欄位值驗證 ────────────────────────────────────────────────────

def validate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """檢查 position 是否為合法值，不合法者設為 None 並記錄警告。"""
    if "position" not in df.columns:
        return df

    invalid_mask = df["position"].notna() & ~df["position"].isin(VALID_POSITIONS)
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        bad = df.loc[invalid_mask, ["name", "position"]].to_dict("records")
        for row in bad:
            logger.warning(
                "位置代碼不合法：%s → '%s'，設為 None",
                row["name"], row["position"],
            )
        df.loc[invalid_mask, "position"] = None

    return df


def validate_gender(df: pd.DataFrame) -> pd.DataFrame:
    """檢查 gender 是否為 M/F。"""
    if "gender" not in df.columns:
        return df

    invalid_mask = ~df["gender"].isin(VALID_GENDERS)
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        logger.error("發現 %d 筆性別欄位異常，將保留原值（需人工確認）", n_invalid)

    return df


# ── 重複偵測 ──────────────────────────────────────────────────────

def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    偵測同隊同背號重複球員，僅記錄警告不刪除。
    同隊同姓名的完全重複則移除。
    """
    # 完全重複列移除
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("移除 %d 筆完全重複列", n_dropped)

    # 同隊同背號警告
    if "jersey_number" in df.columns:
        dup_mask = df.duplicated(
            subset=["team_id", "gender", "jersey_number"], keep=False
        )
        dup_rows = df[dup_mask & df["jersey_number"].notna()]
        if len(dup_rows) > 0:
            for _, row in dup_rows.iterrows():
                logger.warning(
                    "同隊重複背號：%s #%s (%s)",
                    row.get("team_name", "?"),
                    row.get("jersey_number", "?"),
                    row.get("name", "?"),
                )

    return df


# ── 異常值偵測（僅警告） ─────────────────────────────────────────

def flag_outliers(df: pd.DataFrame) -> None:
    """對身高體重超出合理範圍的資料發出警告，不修改資料。"""
    for col, (lo, hi), unit in [
        ("height_cm", HEIGHT_RANGE, "cm"),
        ("weight_kg", WEIGHT_RANGE, "kg"),
    ]:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        outliers = valid[(valid < lo) | (valid > hi)]
        if len(outliers) > 0:
            for idx in outliers.index:
                logger.warning(
                    "異常值：%s %s = %.1f %s（合理範圍 %.0f–%.0f）",
                    df.at[idx, "name"], col, df.at[idx, col],
                    unit, lo, hi,
                )


# ── 資料品質報告 ──────────────────────────────────────────────────

def quality_report(df: pd.DataFrame) -> None:
    """輸出各欄位缺失率統計。"""
    total = len(df)
    print("\n===== 資料品質報告 =====")
    print(f"總筆數：{total}")
    print(f"{'欄位':<20} {'缺失數':>6} {'缺失率':>8}")
    print("-" * 36)
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = n_missing / total * 100 if total > 0 else 0
        print(f"{col:<20} {n_missing:>6} {pct:>7.1f}%")
    print()


# ── 主流程 ────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """執行完整清洗流程，回傳清洗後的 DataFrame。"""
    df = enforce_types(df)
    df = validate_dates(df)
    df = validate_positions(df)
    df = validate_gender(df)
    df = detect_duplicates(df)
    flag_outliers(df)

    # 統一將 NaN 轉為 None（寫入 SQLite 時為 NULL）
    df = df.replace({np.nan: None})

    return df


def main():
    df = load_raw()
    df = clean(df)
    quality_report(df)

    # 輸出清洗後 CSV
    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")
    logger.info("已儲存清洗後資料至 %s", CLEANED_CSV)

    print(f"===== 清洗後前 5 筆 =====")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
