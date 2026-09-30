"""Neueste Version des Bots herunterladen:  python update.py

Ueberschreibt nur Programmdateien. Deine .env (API-Schluessel), state.json,
logs/ und data/ bleiben unberuehrt, weil sie nicht im Download enthalten sind.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://github.com/meronix123/neuralynx/archive/refs/heads/"
       "claude/continue-previous-session-2uhvof.zip")
ROOT = Path(__file__).resolve().parent

print("Lade neueste Version ...")
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
count = 0
for name in z.namelist():
    if "/trading-bot/" not in name or name.endswith("/"):
        continue
    target = ROOT / name.split("/trading-bot/", 1)[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(z.read(name))
    count += 1
print(f"Fertig: {count} Dateien aktualisiert in {ROOT}")
