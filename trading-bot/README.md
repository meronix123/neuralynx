# Bitget Trading-Bot

Handelt BTC, ETH, SOL, XRP, Gold und Silber als Futures auf Bitget,
selbststaendig, mit Stop-Loss und Take-Profit direkt auf der Boerse.

> **Wichtig:** Kein Bot garantiert Gewinne. Bei 10x Hebel reicht eine
> Kursbewegung von ca. 10 % gegen dich fuer den Totalverlust der Position.
> Setze nur Geld ein, dessen Verlust du verkraftest.

## Was der Bot macht

| Teil | Regel |
|---|---|
| Einstieg | 1h-Trend (EMA 50/200) + Ruecksetzer im 5m-Chart (RSI), Volumen und Volatilitaet muessen passen |
| Filter | Keine Trades bei extremer Funding-Rate (ueberhebelter Markt) |
| Stop-Loss | 1,2 x ATR, liegt **auf Bitget** - greift auch, wenn dein PC aus ist |
| Take-Profit | 1,8 x ATR, ebenfalls auf Bitget |
| Gewinne sichern | ab +1R Stop auf Einstand, danach Trailing-Stop (1 x ATR) |
| Risiko je Trade | max. 2 % vom Konto (inkl. Gebuehren) |
| Tageslimit | -6 % am Tag -> keine neuen Trades bis morgen |
| Limits | max. 10 Trades/Tag, max. 2 Positionen gleichzeitig |
| Verlustserie | 3 Verluste in Folge -> 2 Stunden Pause |
| Liquidation | Trades, deren Stop zu nah an der Liquidation laege, werden verworfen |

Alle Werte stehen in `config.yaml` und koennen angepasst werden.

## Installation (Windows)

1. Python 3.11 oder neuer installieren: https://www.python.org/downloads/
   (Haken bei **"Add Python to PATH"** setzen)
2. Diesen Ordner `trading-bot` auf den PC kopieren.
3. PowerShell im Ordner oeffnen und einmalig ausfuehren:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Schritt 1: Backtest (ohne Konto)

```
python run.py backtest --days 60
```

Testet die Strategie an den letzten 60 Tagen echter Bitget-Kurse.
Wichtig ist der **Profit-Faktor**: unter 1,0 verliert die Strategie Geld,
ab ca. 1,3 ist sie brauchbar. Ist er schlecht, **nicht live gehen**.

## Schritt 2: Paper-Modus (echte Kurse, Spielgeld)

In `config.yaml` steht `mode: paper`. Starten:

```
python run.py bot
```

Mindestens 1-2 Wochen laufen lassen und beobachten.

## Schritt 3: Bitget-Demokonto

1. Auf Bitget in den **Demo-Handel** wechseln und dort einen **Demo-API-Schluessel** anlegen.
2. `.env.example` kopieren zu `.env` und Schluessel eintragen.
3. In `config.yaml`: `mode: demo`
4. `python run.py check` (prueft Verbindung), dann `python run.py bot`.

Hier pruefst du, ob Orders, Stop-Loss und Nachziehen des Stops auf Bitget
korrekt ankommen - vor dem ersten echten Euro.

## Schritt 4: Echtgeld

1. Echten API-Schluessel anlegen: **nur Futures-Handel, KEIN Auszahlungsrecht**,
   wenn moeglich auf deine IP beschraenkt.
2. In `.env` eintragen, `mode: live` setzen.
3. `python run.py bot` - der Bot fragt zur Sicherheit nach `JA`.

## Bedienung

- **Stoppen:** `Strg + C`. Offene Positionen behalten Stop-Loss und Take-Profit auf Bitget.
- **Keine neuen Trades, offene weiterlaufen lassen:** leere Datei `STOP` im Ordner anlegen.
- **Protokoll:** `logs/bot.log`
- **Telegram:** `TELEGRAM_TOKEN` und `TELEGRAM_CHAT_ID` in `.env` eintragen.
- Der PC muss laufen, damit neue Trades eroeffnet und Stops nachgezogen werden.
  Energiesparmodus/Ruhezustand deaktivieren.

## Ehrliche Hinweise

- **Gebuehren:** Pro Trade (rein + raus) ca. 0,12 % vom Positionswert. Bei 10x Hebel
  sind das ca. 1,2 % deiner eingesetzten Margin - bei vielen kleinen Trades frisst
  das einen grossen Teil der Gewinne.
- **Slippage:** Bei schnellen Kursbewegungen wird der Stop schlechter ausgefuehrt als eingestellt.
- **Gold/Silber:** Nur wenn Bitget sie als USDT-Futures anbietet; sonst ueberspringt der Bot sie.
- **Mit 30 EUR** ist das Konto sehr klein; bei manchen Paaren kann die Mindest-Ordergroesse
  greifen, dann laesst der Bot den Trade aus.

## Tests

```
pip install pytest
python -m pytest -q
```
