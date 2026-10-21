"""
資料庫備份工具
在任何會改動 data/db/tvl_database.db 結構或內容的遷移前，先備份成帶時間戳的檔案。
"""

import shutil
from datetime import datetime
from pathlib import Path

from src.utils.db_config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def backup_database(db_path: Path | None = None) -> Path:
    """複製 DB 檔到 <db_path>.bak-<timestamp>，回傳備份檔路徑。"""
    source = db_path or DB_PATH
    if not source.exists():
        raise FileNotFoundError(f"找不到要備份的資料庫：{source}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = source.with_name(f"{source.name}.bak-{timestamp}")
    shutil.copy2(source, backup_path)
    logger.info("已備份資料庫：%s -> %s", source, backup_path)
    return backup_path


def main():
    path = backup_database()
    print(f"備份完成：{path}")


if __name__ == "__main__":
    main()
