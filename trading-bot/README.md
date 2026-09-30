# Bitget Trading-Bot

Handelt BTC, ETH, SOL, XRP, Gold und Silber als Futures auf Bitget,
selbststaendig, mit Stop-Loss und Take-Profit direkt auf der Boerse,
Wirtschaftskalender-Filter und Oberflaeche im Browser.

> **Wichtig:** Kein Bot garantiert Gewinne. Bei 10x Hebel reicht eine
> Kursbewegung von ca. 10 % gegen dich fuer den Totalverlust der Position.
> Setze nur Geld ein, dessen Verlust du verkraftest.

## Was der Bot macht

**So entscheidet der Bot:** Zuerst erkennt er je Markt die **Lage**, dann waehlt er die dazu
passende Strategie. Vor dem Einstieg prueft er zusaetzlich Orderbuch und Handelsfluss.

| Marktlage | erkannt an | Strategie |
|---|---|---|
| Trend | ADX >= 22, uebergeordneter Trend klar | Trend-Ruecksetzer: Einstieg nach Ruecksetzer in Trendrichtung (RSI/MACD/EMA-Ausloeser + mind. 5 von 6 Punkten) |
| Seitwaerts | ADX < 18 | Rueckkehr zur Mitte: Kauf am unteren / Verkauf am oberen Bollinger-Band, Ziel = Mitte |
| Ruhephase | Bollinger-Baender so eng wie selten | Ausbruch ueber das 20-Bar-Hoch/-Tief mit 1,5-fachem Volumen |
| Chaos | extreme Volatilitaet | nichts tun |

**Direkt vor jedem Einstieg** (live, nicht im Backtest moeglich):

| Daten | Wirkung |
|---|---|
| Orderbuch (+-0,5 % um den Kurs) | ueberwiegt Kauf- oder Verkaufsdruck? |
| Taker-Fluss (letzte 200 Trades) | kaufen oder verkaufen die aggressiven Marktteilnehmer? |
| beides klar gegen den Trade | kein Einstieg; leicht dagegen -> halbes Risiko |
| Open Interest, Funding | Funding extrem -> kein Trade mit der Masse; OI wird mitgeloggt |
| Wirtschaftskalender | 30 Min vor/nach wichtigen US-Terminen keine neuen Trades |
| Fear & Greed | bei extremer Angst/Gier halbes Risiko |

Alle Messwerte werden bei jedem Trade gespeichert, damit man spaeter auswerten kann,
welche Daten wirklich helfen.

**Zusatz-Filter** (im Backtest und live gleich, einzeln abschaltbar in `config.yaml`):

| Filter | Wirkung |
|---|---|
| Zeitebenen (1m bis 1 Woche) | Richtung auf 1m, 5m, 15m, 1h, 2h, 4h, 1 Tag, 1 Woche. Die hoeheren Zeitebenen ergeben eine Gesamtrichtung (-1..+1); Long nur ab +0,25, Short nur ab -0,25. Die kleinen (1m-15m) werden angezeigt und bei jedem Trade gespeichert. |
| Ueberfuellung (Funding) | Funding-Rate im obersten 10 % des letzten Monats -> zu viele gehebelte Longs -> keine neuen Longs (Short umgekehrt). Rueckwirkend getestet. |
| Makro-Ampel | S&P 500, Nasdaq, US-Dollar, 10-jaehrige US-Zinsen (FRED), Stablecoin-Menge (DefiLlama), BTC-Volatilitaetsindex DVOL (Deribit). Risiko aus -> keine Longs, Risiko an -> keine Shorts. Gold nutzt nur Dollar und Zinsen. Tageswerte mit 2 Tagen Versatz (kein Blick in die Zukunft). |
| Auto-Zeiteinheit | Der Bot beobachtet 15m, 30m, 1h, 2h und 4h gleichzeitig. Jede Zeiteinheit fuehrt ein Schatten-Konto (jedes Signal wird auf dem Papier bis Stop/Ziel verfolgt). Gehandelt wird nur auf Zeiteinheiten, deren letzte 20 Schatten-Trades im Plus sind (Profit-Faktor >= 1). Je Markt immer nur eine Position. `tf_select: fixed` = nur die feste Zeiteinheit. |
| BTC als Leitwaehrung | BTC im Abwaertstrend -> keine Longs bei ETH/SOL/XRP (und umgekehrt) |
| Strategie-Gesundheit | laeuft eine Strategie gerade schlecht (letzte 10 Trades PF < 0,6), wird sie pausiert; nach 10 ausgelassenen Signalen gibt es einen Probe-Trade |
| Zeit-Stop | Trade nach 12 Bars nicht bei +0,5R und kein Teilverkauf -> schliessen |
| ML-Filter (selbstlernend) | lernt aus abgeschlossenen Trades, welche Muster gewinnen, und verwirft Signale mit geschaetzter Gewinnchance unter 45 %. Lernt nur aus der Vergangenheit. Einschalten mit `ml_filter: true`, nachdem `python run.py backtest --days 365` Lernbeispiele gespeichert hat. |

**Absicherung:**

| Teil | Regel |
|---|---|
| Einstieg | Limit-Order (Maker-Gebuehr 0,02 %), wird nach einem Bar ohne Ausfuehrung storniert |
| Stop-Loss | 1 x ATR, liegt **auf Bitget** - greift auch, wenn dein PC aus ist |
| Take-Profit | 2 x ATR, ebenfalls auf Bitget |
| Teilverkauf | bei +1R die Haelfte verkaufen, Stop auf Einstand -> Rest ist risikofrei |
| Risiko je Trade | max. 1 % vom Konto (inkl. Gebuehren) |
| Tageslimit | -6 % am Tag -> keine neuen Trades bis morgen |
| Notbremse | Konto 30 % unter Hoechststand -> Bot eroeffnet gar nichts mehr |
| Limits | max. 3 Positionen, max. 2 in dieselbe Richtung |
| Verlustserie | 4 Verluste in Folge -> 1 Stunde Pause |
| Liquidation | Trades, deren Stop zu nah an der Liquidation laege, werden verworfen |
| Nachkaufen im Verlust | gibt es bewusst NICHT |

Alle Werte stehen in `config.yaml`.

## Wann Echtgeld?

`python run.py report` wertet alle bisherigen Trades aus. Erst wenn im Paper-/Demo-Modus
**mindestens 30 Trades** mit **Profit-Faktor >= 1,3** zusammengekommen sind, sagt er "JA".
Bei 1-2 Trades pro Woche dauert das einige Monate - Abkuerzungen kosten hier meist Geld.

## Oberflaeche

Beim Start von `python run.py bot` laeuft automatisch eine Oberflaeche:
**http://localhost:8050** im Browser oeffnen.

- Chart je Markt mit EMA 21/50, VWAP, Einstieg, Stop-Loss und Take-Profit als Linien
- gestrichelte Linien = wo SL/TP laegen, wenn der Bot jetzt einsteigen wuerde
- "Was der Bot denkt": Trendrichtung, was er erwartet, warum er (nicht) handelt
- alle Indikatoren, kommende Wirtschaftstermine, Fear & Greed
- Liste der letzten Trades mit Ergebnis, Pfeile im Chart

Die Oberflaeche ist nur auf deinem PC erreichbar, nicht aus dem Internet.

## Bitget-Konto in der Oberflaeche

Reiter **Bitget-Konto** in der Oberflaeche: API-Key, Secret und Passphrase eingeben, "Verbinden".
Die Schluessel werden einmal gegen Bitget geprueft und nur lokal in `.env` gespeichert.
Danach siehst du Guthaben (gesamt / verfuegbar / gebunden), alle offenen Positionen mit
Mark-Preis, Liquidation, Margin und unrealisiertem Ergebnis, offene Orders inkl. Stop/Ziel
und die abgeschlossenen Trades der letzten 30 Tage.

Knoepfe: Position schliessen (ganz oder 50 %), Stop-Loss/Take-Profit aendern, Order
stornieren, neue Position eroeffnen (Stop-Loss ist Pflicht), Bot pausieren/fortsetzen.
Achtung: Diese Knoepfe wirken auf das ECHTE Konto (bzw. Demokonto bei "Demo-Schluessel"),
auch wenn der Bot selbst im Paper-Modus laeuft. Die Oberflaeche ist nur auf diesem PC
erreichbar. API-Schluessel immer OHNE Auszahlungsrecht anlegen.

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
python run.py optimize --days 365
```

Rechnet ca. 580 Varianten durch: Zeiteinheit (1h/4h), welche Strategien erlaubt sind
(nur Trend / nur Seitwaerts / nur Ausbruch / alle je nach Marktlage), Punkte-Schwelle,
Stop-Abstand, Chance/Risiko, Market- oder Limit-Einstieg, Teilverkauf ja/nein und
Zusatz-Filter (ohne / BTC+Gesundheit+Zeit-Stop / zusaetzlich ML-Filter).
Zeigt fuer die beste Variante auch, wie jede Strategie einzeln abgeschnitten hat.

Ehrlicher Test: Ausgewaehlt wird auf den ersten 2/3 der Daten, bewertet auf dem
letzten 1/3, das die Auswahl nie gesehen hat. Nur Varianten, die in **beiden**
Zeitraeumen profitabel sind, kommen in Frage. Alle Ergebnisse stehen danach in
`data\optimierung.csv` (mit Excel oeffnen).

Nachkaufen im Verlust ist bewusst NICHT eingebaut - mit Hebel der schnellste Weg
zum Totalverlust. Aufgestockt wird nur, wenn der Trade im Gewinn ist, und der
Stop wird danach mindestens auf den neuen Einstand gezogen.

### Feste Zeiteinheit oder automatische Wahl?

```
python run.py optimize --zeiten
```

Vergleicht auf 365 Tagen (15m-Daten): fest 15m / 30m / 1h / 2h / 4h, alle gleichzeitig
und die automatische Wahl (verschieden streng). Zeigt, wie viele Trades pro Tag jeder
Modus macht und ob er im Test-Zeitraum Geld verdient. Ergebnisse in `data\zeiteinheiten.csv`.

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
