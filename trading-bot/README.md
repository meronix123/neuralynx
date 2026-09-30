# Bitget Trading-Bot

Handelt BTC, ETH, SOL, XRP, Gold und Silber als Futures auf Bitget,
selbststaendig, mit Stop-Loss und Take-Profit direkt auf der Boerse,
Wirtschaftskalender-Filter und Oberflaeche im Browser.

> **Wichtig:** Kein Bot garantiert Gewinne. Bei 10x Hebel reicht eine
> Kursbewegung von ca. 10 % gegen dich fuer den Totalverlust der Position.
> Setze nur Geld ein, dessen Verlust du verkraftest.

## Was der Bot macht

**Vor jedem Trade prueft er:**

| Bereich | Was geprueft wird |
|---|---|
| Trend | 1h-Chart: EMA 50 ueber/unter EMA 200 - gehandelt wird nur in Trendrichtung |
| Ausloeser | RSI dreht (ueber 40 / unter 60), MACD-Histogramm kreuzt 0, oder Kurs erobert EMA 21 zurueck |
| Punkte (min. 4 von 6) | Struktur (EMA 21/50), MACD, ADX-Trendstaerke, Volumen, nicht ueberkauft/-verkauft (RSI + Bollinger), Kurs ueber/unter VWAP |
| Volatilitaet | ATR darf nicht zu klein (Gebuehren fressen alles) und nicht zu gross sein |
| Funding-Rate | extrem einseitig gehebelter Markt -> kein Trade in dieselbe Richtung |
| Wirtschaftsdaten | holt selbst den Wirtschaftskalender: 30 Min vor bis 30 Min nach wichtigen US-Terminen (Zinsentscheid, Inflation, Arbeitsmarkt ...) keine neuen Trades |
| Stimmung | Crypto Fear & Greed Index: bei extremer Angst/Gier halbes Risiko |
| Spread | zu grosse Geld/Brief-Spanne -> kein Trade |
| Gold/Silber | am Wochenende pausiert (duenner Markt) |

**Absicherung:**

| Teil | Regel |
|---|---|
| Stop-Loss | 1,2 x ATR, liegt **auf Bitget** - greift auch, wenn dein PC aus ist |
| Take-Profit | 1,8 x ATR, ebenfalls auf Bitget |
| Gewinne sichern | ab +1R Stop auf Einstand, danach Trailing-Stop (1 x ATR) |
| Risiko je Trade | max. 1 % vom Konto (inkl. Gebuehren) |
| Tageslimit | -6 % am Tag -> keine neuen Trades bis morgen |
| Notbremse | Konto 30 % unter Hoechststand -> Bot eroeffnet gar nichts mehr |
| Limits | max. 30 Trades/Tag, max. 3 Positionen, max. 2 in dieselbe Richtung |
| Verlustserie | 4 Verluste in Folge -> 1 Stunde Pause |
| Liquidation | Trades, deren Stop zu nah an der Liquidation laege, werden verworfen |

Alle Werte stehen in `config.yaml`. Mehr Trades: `min_score: 3`, weniger: `min_score: 5`.

## Oberflaeche

Beim Start von `python run.py bot` laeuft automatisch eine Oberflaeche:
**http://localhost:8050** im Browser oeffnen.

- Chart je Markt mit EMA 21/50, VWAP, Einstieg, Stop-Loss und Take-Profit als Linien
- gestrichelte Linien = wo SL/TP laegen, wenn der Bot jetzt einsteigen wuerde
- "Was der Bot denkt": Trendrichtung, was er erwartet, warum er (nicht) handelt
- alle Indikatoren, kommende Wirtschaftstermine, Fear & Greed
- Liste der letzten Trades mit Ergebnis, Pfeile im Chart

Die Oberflaeche ist nur auf deinem PC erreichbar, nicht aus dem Internet.

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

## Schritt 1b: Strategie-Tester (beste Einstellungen finden)

```
python run.py optimize --days 180
```

Rechnet ca. 1300 Varianten durch: Zeiteinheit (15m/1h/4h), Punkte-Schwelle,
Stop-Abstand, Chance/Risiko, Trailing an/aus, Market- oder Limit-Einstieg und
Positionsfuehrung (normal / Teilverkauf bei +1R / Aufstocken im Gewinn / beides).

Ehrlicher Test: Ausgewaehlt wird auf den ersten 2/3 der Daten, bewertet auf dem
letzten 1/3, das die Auswahl nie gesehen hat. Nur Varianten, die in **beiden**
Zeitraeumen profitabel sind, kommen in Frage. Alle Ergebnisse stehen danach in
`data\optimierung.csv` (mit Excel oeffnen).

Nachkaufen im Verlust ist bewusst NICHT eingebaut - mit Hebel der schnellste Weg
zum Totalverlust. Aufgestockt wird nur, wenn der Trade im Gewinn ist, und der
Stop wird danach mindestens auf den neuen Einstand gezogen.

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
  sind das ca. 1,2 % deiner eingesetzten Margin. Bei 30 Trades am Tag ist das der
  groesste Kostenblock - der Backtest rechnet die Gebuehren mit ein.
- **Backtest** simuliert Wirtschaftskalender, Fear & Greed und Funding nicht; die
  Ergebnisse live koennen deshalb abweichen.
- **Slippage:** Bei schnellen Kursbewegungen wird der Stop schlechter ausgefuehrt als eingestellt.
- **Gold/Silber:** Nur wenn Bitget sie als USDT-Futures anbietet; sonst ueberspringt der Bot sie.
- **Mit 30 EUR** ist das Konto sehr klein; bei manchen Paaren kann die Mindest-Ordergroesse
  greifen, dann laesst der Bot den Trade aus.

## Tests

```
pip install pytest
python -m pytest -q
```
