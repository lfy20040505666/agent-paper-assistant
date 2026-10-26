"""项目路径和 .env 读取（不依赖 python-dotenv）。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"   # 下载的 PDF 和转出的文字
NOTES_DIR = PROJECT_ROOT / "notes"   # Agent 保存的笔记
TRACE_DIR = PROJECT_ROOT / "runs"    # 每次运行的逐步记录


def load_dotenv(path=PROJECT_ROOT / ".env"):
    """把 KEY=VALUE 形式的行读进环境变量；已经存在的环境变量优先。"""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
