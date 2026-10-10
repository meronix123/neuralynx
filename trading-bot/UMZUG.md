# Trading-Bot: Umzug auf einen neuen PC

Stand: 10.10.2026. Alles, was du brauchst, um auf dem neuen PC genau dort weiterzumachen, wo du jetzt bist.

---

## 0. Zwei Dinge vorab

**Der Chat mit Claude liegt in der Cloud, nicht auf deinem PC.** Auf dem neuen PC einfach claude.ai (oder die
Claude-App, Bereich "Code") mit demselben Konto oeffnen und die bisherige Sitzung waehlen.

**Der Bot darf nie auf zwei PCs gleichzeitig laufen.** Beide wuerden dasselbe Bitget-Konto handeln, doppelte
Positionen eroeffnen und sich gegenseitig Stops loeschen. Also: erst auf dem alten PC stoppen, dann auf dem
neuen starten.

---

## 1. Was vom alten PC mitmuss

Der Programmcode kommt immer frisch von GitHub. Mitnehmen musst du nur deine **eigenen Daten** aus `~/trading-bot`:

| Datei / Ordner | Inhalt | noetig? |
|---|---|---|
| `.env` | Bitget-API-Schluessel (Echt- und Testkonto), Modus (`BOT_MODE`), `BOT_LIVE_CONFIRMED=1`, Fernzugriff-Passwort (nur als Hash), ggf. Telegram | **ja** |
| `data/` | KI-Gedaechtnis (Kerzen aller Maerkte), gespeicherte KI-Modelle, gepruefte KI-Entscheidungen (Live-Trefferquote), Trade-Verlauf (`sitzungen_*.jsonl`), laufende Autopilot-Sitzung mit offenen Positionen (`sitzung_ki_*.json`), Bot-Zustand | **ja** |
| `state.json` | Zustand des Regel-Bots (Hoechststand, Tagesstart) | ja, falls vorhanden |
| `config.yaml` | deine Aenderungen an den Einstellungen | ja |
| `logs/` | Protokolle | nein, nur zum Nachlesen |

**Nicht** mitnehmen: `.venv` (wird neu gebaut), `bot/`, `tests/` (kommen von GitHub).

Die KI-Einstellungen im Browser (Formular im KI-Feld) merkt sich nur der jeweilige Browser. Laeuft die
Autopilot-Sitzung auf dem neuen PC weiter, fuellt das Formular sich automatisch mit den laufenden Werten.

---

## 2. Alter PC: stoppen und einpacken

Terminal oeffnen und Schritt fuer Schritt:

**a) Bot stoppen und Autostart abschalten** (offene Positionen bleiben auf Bitget mit Stop und Ziel bestehen,
die Sitzung ist in `data/` gespeichert und wird auf dem neuen PC fortgesetzt):

```
sudo systemctl stop trading-bot && sudo systemctl disable trading-bot && systemctl is-active trading-bot
```

Die letzte Zeile muss `inactive` zeigen.

**b) Deine Daten in eine Datei packen:**

```
cd ~/trading-bot && tar -czf ~/trading-bot-umzug.tar.gz $(ls -d .env config.yaml state.json data logs 2>/dev/null) && ls -lh ~/trading-bot-umzug.tar.gz && tar -tzf ~/trading-bot-umzug.tar.gz | grep -E "^\.env$|^data/$|sitzung" | head
```

Es muss `.env`, `data/` und (bei laufendem Autopilot) eine `sitzung_ki_...json` erscheinen.

**c) Auf einen USB-Stick kopieren:** Stick einstecken, im Dateimanager `trading-bot-umzug.tar.gz` aus deinem
Home-Ordner auf den Stick ziehen. Oder im Terminal (Stick-Name anpassen, `ls /media/$USER` zeigt ihn):

```
cp ~/trading-bot-umzug.tar.gz /media/$USER/NAME_DES_STICKS/
```

> **Achtung:** Die Datei enthaelt deine API-Schluessel. Nicht in eine Cloud, nicht in einen Chat, nicht nach
> GitHub laden. Nach dem Umzug vom Stick loeschen.

---

## 3. Neuer PC (Ubuntu / Linux)

**a) Grundpakete:**

```
sudo apt update && sudo apt install -y python3 python3-venv python3-pip curl
```

**b) Programm von GitHub holen** (landet in `~/trading-bot`):

```
cd ~ && curl -L -o nx.tar.gz https://github.com/meronix123/neuralynx/archive/refs/heads/claude/continue-previous-session-2uhvof.tar.gz && rm -rf nx_tmp && mkdir nx_tmp && tar -xzf nx.tar.gz -C nx_tmp --strip-components=1 && mkdir -p ~/trading-bot && cp -r nx_tmp/trading-bot/. ~/trading-bot/ && rm -rf nx_tmp nx.tar.gz && ls ~/trading-bot
```

**c) Python-Umgebung bauen** (dauert 1 bis 3 Minuten):

```
cd ~/trading-bot && python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
```

**d) Deine Daten vom Stick einspielen** (Stick-Name anpassen):

```
cp /media/$USER/NAME_DES_STICKS/trading-bot-umzug.tar.gz ~/ && tar -xzf ~/trading-bot-umzug.tar.gz -C ~/trading-bot && ls -la ~/trading-bot/.env ~/trading-bot/data | head
```

**e) Verbindung zu Bitget pruefen:**

```
cd ~/trading-bot && .venv/bin/python run.py check
```

Muss die Verbindung und die Schluessel als in Ordnung melden.

**f) Als Dienst einrichten** (startet automatisch mit dem PC, neu nach Absturz). Der Befehl setzt deinen
Benutzernamen und Pfad selbst ein:

```
sudo tee /etc/systemd/system/trading-bot.service >/dev/null <<EOF
[Unit]
Description=Bitget Trading-Bot
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$HOME/trading-bot
ExecStart=$HOME/trading-bot/.venv/bin/python run.py bot
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now trading-bot && sleep 20 && journalctl -u trading-bot -n 15 --no-pager
```

Im Log sollte stehen, dass der Bot laeuft und der KI-Autopilot "nach Neustart fortgesetzt" wurde.

**g) Oberflaeche oeffnen:** im Browser `http://localhost:8050`.
- Reiter "Bitget-Konto": muss verbunden sein (Schluessel kommen aus der `.env`).
- KI-Feld: Uebersicht zeigt "AN" und "nach Neustart fortgesetzt", offene Positionen stehen in der Tabelle.
- Die KI-Modelle sind mitgekommen, die KI handelt sofort. Das gemeinsame Pool-Modell baut sich in 15 Minuten neu auf.

**h) Fernzugriff vom Handy** (falls genutzt): Tailscale auf dem neuen PC einrichten, die Handy-Adresse aendert sich.

```
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up && tailscale ip -4
```

Auf dem Handy dann `http://<diese-IP>:8050` oeffnen. Das Passwort bleibt dasselbe (steckt in der `.env`).

---

## 4. Neuer PC mit Windows (falls kein Linux)

1. Python 3.11 oder neuer installieren: https://www.python.org/downloads/ (Haken bei "Add Python to PATH").
2. Code als ZIP laden und nach `C:\trading-bot` entpacken (nur den Unterordner `trading-bot`):
   https://github.com/meronix123/neuralynx/archive/refs/heads/claude/continue-previous-session-2uhvof.zip
3. PowerShell in `C:\trading-bot` oeffnen:

   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   tar -xzf E:\trading-bot-umzug.tar.gz -C C:\trading-bot
   .venv\Scripts\python run.py check
   .venv\Scripts\python run.py bot
   ```

   (`E:` durch den Laufwerksbuchstaben des Sticks ersetzen.) Das Fenster muss offen bleiben, solange der Bot
   laufen soll. Fuer Autostart: Aufgabenplanung, Aufgabe "Beim Anmelden", Programm
   `C:\trading-bot\.venv\Scripts\python.exe`, Argumente `run.py bot`, Starten in `C:\trading-bot`.

---

## 5. Danach

**Spaetere Updates (Linux), wie bisher:**

```
cd ~ && curl -L -o nx.tar.gz https://github.com/meronix123/neuralynx/archive/refs/heads/claude/continue-previous-session-2uhvof.tar.gz && rm -rf nx_tmp && mkdir nx_tmp && tar -xzf nx.tar.gz -C nx_tmp --strip-components=1 && cp -r nx_tmp/trading-bot/. ~/trading-bot/ && rm -rf nx_tmp nx.tar.gz && sudo systemctl restart trading-bot && sleep 20 && journalctl -u trading-bot -n 5 --no-pager
```

**Alten PC aufraeumen**, sobald der neue laeuft (Bot ist dort schon gestoppt und abgeschaltet):

```
rm ~/trading-bot/.env ~/trading-bot-umzug.tar.gz
```

Damit liegen deine Schluessel nicht mehr auf dem alten Rechner. Den USB-Stick ebenfalls leeren.

**Wichtige Befehle auf einen Blick:**

| Zweck | Befehl |
|---|---|
| Status | `systemctl status trading-bot --no-pager` |
| Log live | `journalctl -u trading-bot -f` (Strg+C beendet) |
| Neustart | `sudo systemctl restart trading-bot` |
| Stopp | `sudo systemctl stop trading-bot` |
| Tagesbilanz | `journalctl -u trading-bot --since "24 hours ago" --no-pager \| grep -E "netto\|Tagesverlust" \| tail -40` |
| KI-Bericht | `cd ~/trading-bot && .venv/bin/python run.py ki-bericht \| head -80` |

---

## 6. Stand des Projekts (Kurzfassung)

- **Ein System:** nur der KI-Autopilot handelt. Die Regel-Strategie, Orderfluss, Muster, Makro und Zeitebenen
  sind Eingaben der KI ("Supergehirn").
- **KI:** 5-Minuten-Modell (bis 30 min) und Turbo-1-Minuten-Modell, gemeinsames Lernen ueber alle Maerkte
  (Pool), ehrliche Kalibrierung und Live-Bilanz je Markt, Modelle bleiben ueber Neustarts erhalten.
- **KI-Feld (vier Abschnitte):** Wo handeln, Methode (Tempo, KI-Sicherheit, Selbst-Anpassung, Netz), Groesse &
  Risiko (max. 3 Positionen, Gesamt-Risiko, Tagesverlust), Schutz-Sperren (alle einzeln abschaltbar, auf dem
  echten Konto mit Bestaetigung OHNE SCHUTZ).
- **Selbst-Anpassung:** bringt eine Methode 20 Std netto Verlust oder gar keinen Trade, wechselt die KI selbst.
- **Bisherige Ergebnisse:** Turbo und Netz waren live netto negativ (Gebuehren). Am meisten Vorsprung hat die
  KI im Tempo "Normal" (15-30 min) auf Maerkten, auf denen sie live ueber 53 % trifft.
- **Empfohlene Einstellung:** Tempo Normal, KI-Sicherheit 56 %, Groesse 1 % Risiko, max. 3 Positionen,
  Gesamt-Risiko 6 %, Tagesverlust 5 %, Netz aus, Kosten-Schutz an, Selbst-Anpassung an.
- **Code:** GitHub `meronix123/neuralynx`, Branch `claude/continue-previous-session-2uhvof`, Ordner `trading-bot`.
