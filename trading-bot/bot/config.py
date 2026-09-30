import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    load_dotenv(ROOT / ".env")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["mode"] not in ("paper", "demo", "live"):
        raise ValueError(f"Unbekannter mode: {cfg['mode']}")
    cfg["api"] = {
        "key": os.getenv("BITGET_API_KEY", ""),
        "secret": os.getenv("BITGET_API_SECRET", ""),
        "password": os.getenv("BITGET_API_PASSPHRASE", ""),
    }
    cfg["telegram"] = {
        "token": os.getenv("TELEGRAM_TOKEN", ""),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    }
    return cfg
