import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    load_dotenv(ROOT / ".env")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # In der Oberflaeche gewaehlter Modus (steht in .env, damit ein Update ihn nicht zuruecksetzt)
    cfg["mode"] = os.getenv("BOT_MODE") or cfg["mode"]
    if cfg["mode"] not in ("paper", "demo", "live"):
        raise ValueError(f"Unbekannter mode: {cfg['mode']}")
    prefix = "BITGET_DEMO_API_" if cfg["mode"] == "demo" and os.getenv("BITGET_DEMO_API_KEY") else "BITGET_API_"
    cfg["api"] = {
        "key": os.getenv(prefix + "KEY", ""),
        "secret": os.getenv(prefix + "SECRET", ""),
        "password": os.getenv(prefix + "PASSPHRASE", ""),
    }
    cfg["telegram"] = {
        "token": os.getenv("TELEGRAM_TOKEN", ""),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    }
    return cfg
