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
| Order Blocks | letzte Gegenkerze vor einer kraeftigen Bewegung (mind. 1,5 x ATR) = Zone, an der der Kurs oft reagiert. Im Chart gruen (Kauf-Zone) / rot (Verkaufs-Zone). Filter (`ob_filter`): avoid = keine Trades, wenn eine Gegen-Zone den Weg zum Ziel versperrt; confirm = nur nach Antest einer passenden Zone. Standard aus - erst mit `python run.py optimize --orderblocks` pruefen. |
| Orderbuch-Waende | grosse Kauf-/Verkaufsauftraege (mind. 3 x so viel wie ueblich, +-2 % um den Kurs), live im Chart und in der Markt-Tabelle. Optionaler Filter `wall_filter`. Nicht rueckwirkend testbar. |
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

Oben links waehlst du, womit der **Bot** handelt: SIMULATION (Paper), TESTKONTO (Bitget-Demo)
oder ECHTES KONTO (nur nach Eingabe von JA). Der Bot startet dann selbst neu. Jeder Modus hat
seine eigene Trade-Historie (state.json, state_demo.json, state_live.json).
Test- und Echtkonto haben getrennte Schluessel; im Reiter Bitget-Konto kannst du zwischen
beiden umschalten. "Bot AN/AUS" stoppt neue Trades, "Alle Positionen schliessen" gibt es
fuer den Bot und fuer das Bitget-Konto.

Knoepfe: Position schliessen (ganz oder 50 %), Stop-Loss/Take-Profit aendern, Order
stornieren, neue Position eroeffnen (Stop-Loss ist Pflicht), Bot pausieren/fortsetzen.
Achtung: Diese Knoepfe wirken auf das ECHTE Konto (bzw. Demokonto bei "Demo-Schluessel"),
auch wenn der Bot selbst im Paper-Modus laeuft. Die Oberflaeche ist nur auf diesem PC
erreichbar. API-Schluessel immer OHNE Auszahlungsrecht anlegen.

## KI-Prognose (30 Minuten)

In beiden Charts zeigt eine tuerkise, gestrichelte Linie, wohin der Kurs in den naechsten
30 Minuten laufen koennte, dazu gepunktet ein 80-%-Band (dort sollte der Kurs in 8 von 10 Faellen
landen). Gelernt wird je Markt aus den letzten ~1000 5-Minuten-Kerzen (Renditen, RSI, EMAs, Bollinger,
MACD, Volumen, Volatilitaet, Tageszeit), alle 5 Minuten neu.

Ehrlich: Kurzfristige Kurse sind zum grossen Teil Zufall. Das Modell wird deshalb nur mit den
aelteren 80 % der Daten gelernt und an den neuesten 20 % geprueft. Ist es dort nicht besser als
"Kurs bleibt gleich", ist die Linie **grau** und es steht "keine echte Vorhersagekraft" dabei.
Unter dem Chart steht ausserdem die **Live-Trefferquote**: Jede Prognose wird nach 30 Minuten mit
dem echten Kurs verglichen. Erst wenn die ueber viele Prognosen deutlich ueber 50 % liegt, hat
die Prognose einen Wert. Der Bot handelt NICHT nach dieser Prognose. Abschalten: Haken "KI-Prognose" im Chart.

## Speed-Trading

Im Reiter Bot unter dem Chart: Maerkte, Dauer (1-30 min), Einsatz, Hebel, Tempo und Konto
(Simulation oder das im Reiter Bitget-Konto verbundene Konto) waehlen, Start. Der Bot handelt dann
kleine Bewegungen: Signal aus 1/2/3/5-Minuten-Chart und Orderbuch, Einstieg per Post-Only-Limit-Order
(Maker), Ziel als Limit-Order, Stop als an die Position gebundener Bitget-Stop, je Trade hoechstens
5 Minuten. Nach Ablauf (oder "stoppen") werden offene Speed-Positionen geschlossen. Live-Anzeige mit
Netto-Ergebnis NACH Gebuehren. Ehrlich: Bei so kleinen Bewegungen fressen Gebuehren viel - erst in
der Simulation und auf dem Testkonto pruefen, ob netto etwas uebrig bleibt.

## KI-Bericht

```
python run.py ki-bericht
```
Wertet die KI auf diesem PC aus: gepruefte Live-Entscheidungen je Markt (gesamt, je Sicherheitsstufe),
Test-Trefferquote je Tag, frischer Test mit den gespeicherten Kerzen je Vorhersagezeit (5-30 min) und
wie viele Trades pro Tag der Autopilot bei 54/56/58/60 % Mindest-Sicherheit gemacht haette - und wie
oft er dann richtig lag.

KI-Autopilot "Fast": Vorhersage 5-10 min, enge Stops/Ziele aus der 5-Minuten-Schwankung, Pruefung alle
1,5 s - viele Trades, aber auch viele Gebuehren. Nur sinnvoll, wenn der KI-Bericht auf 5-10 min eine
Trefferquote klar ueber 55 % zeigt.

KI-Autopilot "Turbo" (Hebel-Speed-KI): eigenes 1-Minuten-Modell (lernt nur fuer 1-5 min voraus,
eigener Speicher `data/ki1m`), Pruefung jede Sekunde, Stop aus der 1-Minuten-Schwankung. Mit
Groesse "auto" bestimmt die KI Einsatz und Hebel selbst: Risiko je Trade waechst mit der Sicherheit,
der Hebel wird so gewaehlt, dass der Liquidationspreis mindestens doppelt so weit weg ist wie der
Stop, und nie ueber "max. Hebel". Hebel macht Trades nicht besser - er vergroessert nur Gewinn UND
Verlust. Der KI-Bericht zeigt am Ende einen eigenen Abschnitt "TURBO-KI".

Groesse im Autopilot: `auto` (KI waehlt), `risk` (x % des Kontos Verlust bis Stop), `pct` (x % des
Kontos als Einsatz), `usdt` (fester Einsatz). Teilkauf = erst ein Teil, Rest nur nachkaufen, wenn die
Position im Gewinn ist (nie im Verlust). Teilverkauf % = wie viel beim ersten Ziel verkauft wird.

Live aufgezeichnet und gelernt (Bitget liefert das nicht rueckwirkend, die KI sammelt es selbst je Kerze):
Orderbuch-Druck und -Waende, **Open Interest** (Positionsaufbau/-abbau, auch mit der Kursrichtung
kombiniert: OI rauf + Kurs rauf = neue Longs, OI runter + Kurs rauf = Eindeckung), **Taker-Fluss**
(aggressive Kaeufe minus Verkaeufe) und **Basis** (Mark- minus Indexpreis). Diese Merkmale greifen,
sobald etwa 800 Kerzen (knapp 3 Tage) aufgezeichnet sind - der Ordner `data` muss dafuer erhalten bleiben.

## Was die KI gelernt hat - aus der Forschung (Stand Oktober 2026)

Vier Recherchen (Mikrostruktur, Lernmethodik, Indikatoren/Strategien, Nachrichten/Stimmung; ueber 150
Quellen, bevorzugt begutachtete Studien) wurden ausgewertet. Nur Belegtes ist eingebaut:

- **Viertelstunden-Effekt** (3 Studien, auch ausserhalb der Stichprobe): Bewegung und Volumen konzentrieren
  sich auf die Minuten :00/:15/:30/:45; der Taker-Fluss direkt nach der Marke sagt die naechsten Stunden
  voraus. Merkmale: Kerze enthaelt Stunden-/Viertelstunden-Marke, Taker-Fluss nach der Marke (live aufgezeichnet).
- **Handelsfenster** statt glatter Uhrzeit: US-Eroeffnung 13:30 UTC, US-Schluss, 21-23 UTC (beste BTC-Stunden),
  02-05 UTC (tot), Wochenende, Funding 00/08/16 UTC; Gold/Silber: COMEX-Eroeffnung/-Schluss, London-Fixing
  (Londoner Zeit), Ueberlappung London/New York.
- **Momentum nur bei hohem Volumen/hoher Schwankung, sonst Rueckkehr zum Mittel** - das Kernergebnis der
  Krypto-Intraday-Forschung: Wechselwirkungen Rendite x Volumen-Z-Wert, Rendite x Volatilitaets-Regime,
  Varianz-Verhaeltnis, ADX, VWAP-Abstand, Ertrag seit Tagesbeginn/US-Eroeffnung.
- **Volatilitaet**: Garman-Klass (schlaegt GARCH bei Krypto), Verhaeltnis kurz/lang, Dochtigkeit, Bollinger-Breite.
- **Basis Letztkurs/Index** (theoretisch verankert, mean-reverting zum Funding) - ueber Bitgets Index-Kerzen fuer
  die ganzen 70 Tage rueckwirkend geladen; dazu Open Interest und Taker-Fluss (live).
- **Zielwerte wie die Handelsregel** (Triple-Barrier, in BTC/ETH-Studien nach Kosten besser): ein zweiter
  Lernkopf je Richtung lernt "Wird das 2-R-Ziel VOR dem 1-R-Stop erreicht (innerhalb 4 x Prognosezeit)?",
  Platt-kalibriert. Der Autopilot handelt nur bei positivem Erwartungswert nach Kosten und bestimmt die
  Groesse nach Viertel-Kelly (Deckel: Risiko %). Der KI-Bericht zeigt beides je Markt.
- **Ehrlichkeit**: Vertrauen in die Richtung erst ab 2 Standardfehlern Vorsprung und nie hoeher als die
  Trefferquote in den 20 % staerksten Momenten; staerkere Regularisierung zur Auswahl (viele Merkmale,
  wenig Daten).
- **Ausstiege** (Studienlage): Einstand erst ab +1,3 R; ATR-Nachzieh-Stop nur im Trend (ADX >= 20) oder bei
  Erschoepfung; Zeit-Ausstiege wie bisher.
- **Not-Aus**: stimmt die versprochene Sicherheit ueber 20 Trades nicht mit den Ergebnissen ueberein
  (Log-Loss-Vergleich), pausiert der Markt 12 h - ohne Neu-Training (Drift-Detektoren als Retrain-Ausloeser
  sind laut Studien eine Illusion).
- **Schutz vor Boersen-Anomalien** (dokumentierte Bitget-Vorfaelle: zurueckgerollte Trades, Flash-Wicks,
  Wartungen): kein Einstieg bei Wartungsstatus, zu weitem Spread, Kurssprung > 4 Sigma in einer Minute (10 min
  Pause) oder Letztkurs > 0,3 % vom Mark-Preis entfernt; keine Markt-Einstiege um :00/:15/:30/:45 und Funding.

Bewusst NICHT eingebaut (keine oder negative Belege fuer 5-30 min): Fibonacci-Level, harmonische Muster,
Elliott-Wellen, Wyckoff, Ichimoku, Supertrend, Heikin-Ashi, Stochastik/CCI/Williams, Pivot-Punkte, Liquidity
Sweeps als Umkehrsignal, Long/Short-Verhaeltnis, Twitter/Reddit-Stimmung, Google Trends, ETF-Fluesse als
Intraday-Signal, Whale-Alerts, Polymarket/FedWatch (laufen dem Kurs hinterher), Hurst-Regimefilter, LSTM/
Transformer auf 20k Kerzen. Realistische Erwartung laut ehrlichen Studien: wenige, selektive Trades mit
kleinem Vorteil - keine hohe Trefferquote.

## Kosten-Schutz (warum die KI nicht staendig handelt)

Rechnung mit Bitget-Gebuehren (0,02 % Maker, 0,06 % Taker, dazu Schlupf beim Stop): Bei 1-5 min
Vorhersage verliert jede Strategie, selbst wenn die KI 60 % trifft - die Bewegungen sind kleiner als die
Kosten. Ins Plus kommt sie nur bei 20-30 min Vorhersage ab ~57 % Treffern (10-15 min: ~60 %), mit einem
Stop von mind. 4x den Kosten eines Trades. Der Autopilot handelt deshalb nur solche Trades:

- Mindest-Sicherheit je Vorhersagezeit (57 % / 60 %), 1-5 min nur mit live nachgewiesenen 60 % Treffern
- Stop mind. 4x die Kosten; Einstand erst ab +1 R, Teilverkauf ab +1,5 R (frueher schnitt das Gewinner ab)
- Ausstieg bei KI-Wende nur bei deutlich hoeherer Sicherheit und nicht direkt nach dem Einstieg
- Limit nicht gefuellt, KI weiter dafuer -> Einstieg zum Marktpreis (sonst fuellen sich fast nur die
  Verlierer-Trades)
- Lernt aus den eigenen Abschluessen: verliert ein Markt in den letzten 3 Tagen dauerhaft
  (Gewinnfaktor unter 0,7 bei mind. 12 Trades), pausiert er dort

Gewinne sichern (laeuft nicht mehr endlos im Plus):

- ab +1 R Stop auf Einstand; danach nie mehr als die Haelfte des besten Gewinns zurueckgeben (ab +2 R
  hoechstens 30 %)
- Nachzieh-Stop mit ATR der 5-Minuten-Kerzen (2,5 x ATR, mind. 0,8 R Abstand)
- Erschoepfung erkannt (RSI ueber 72 / unter 28, Umkehrkerze wie Shooting Star oder Engulfing,
  EMA 9 unter 21): Stop enger (1,2 x ATR, 70 % des Gewinns bleiben)
- Prognosezeit doppelt vorbei und KI nicht mehr dafuer: Gewinn mitnehmen; viermal vorbei ohne klaren
  Gewinn: schliessen
- Ziel-Order fehlt auf Bitget: der Autopilot schliesst beim Ziel selbst

Margin, Hebel und Liquidation werden aus den echten Bitget-Werten angezeigt; der Hebel wird nach dem
Setzen bei Bitget nachgelesen und die Groesse damit gerechnet.

Sicherheit und Dauerbetrieb:

- Stop laesst sich nicht setzen: Position bleibt als "ungeschuetzt" markiert, wird jede Runde erneut
  abgesichert und nach 60 s zur Sicherheit geschlossen; ein Markt-Einstieg wird nie doppelt gekauft
- Neustart des Bots (Update, Absturz): Konto-Sitzungen werden von Platte fortgesetzt (`data/sitzung_*.json`),
  offene Positionen auf Bitget geprueft und weiter gefuehrt; wartende Limit-Orders werden storniert
- Tagesverlust-Limit (5 %): keine neuen Trades bis zum naechsten Tag (UTC), die Sitzung bleibt an
- Gesamt-Risiko: alle offenen Positionen zusammen hoechstens 3 % des Kontos am Stop (Krypto laeuft
  meist gemeinsam - sechs Longs sind EINE Wette)
- Bitget-Ratenlimit oder Netzausfall: 20 s bzw. 5 s Pause statt Dauerfeuer (sonst IP-Sperre)
- Hoechstens 2 KI-Trainings gleichzeitig; der Entscheidungs-Speicher haelt 20 000 gepruefte Prognosen
  und wird nie halb geschrieben

Auf dem Bitget-Konto ist der Kosten-Schutz immer an, in der Simulation abschaltbar (zum Vergleichen).

## Handelszeiten Gold/Silber

Gold und Silber sind Freitag 21:00 bis Sonntag 22:00 UTC und taeglich 21:00-22:00 UTC zu (anpassbar in
`config.yaml` unter `hours`). 30 min vor Schluss keine neuen Positionen, Speed/Autopilot schliessen
Metall-Positionen kurz vor dem Wochenende. Die KI lernt nicht aus Kerzen, in denen der Markt still stand.

## Handelsbericht

```
python run.py handelsbericht
```
Wertet alle Trades aus (Bitget-Verlauf und die Protokolle `data/sitzungen_*.jsonl`): Ergebnis je Markt,
Richtung, Tageszeit, Haltedauer, Ausstiegsgrund und KI-Sicherheit, inkl. Gebuehren-Anteil. Markiert,
wo dauerhaft Geld verloren geht.

## Vom Handy aus ansehen (Fernzugriff)

1. Auf dem PC: `python run.py fernzugriff` - Passwort festlegen (mind. 10 Zeichen, wird nur als Hash gespeichert).
2. Fuer unterwegs **Tailscale** einrichten (kostenlos, privates verschluesseltes Netz - KEINE Portfreigabe im Router!):
   - PC (Linux): `curl -fsSL https://tailscale.com/install.sh | sh` und `sudo tailscale up` (Link im Browser bestaetigen)
   - Handy: App "Tailscale" installieren, mit demselben Konto anmelden
3. Bot neu starten. Er zeigt die Handy-Adresse an (z. B. `http://100.x.y.z:8050`), alternativ `tailscale ip -4`.
4. Auf dem Handy oeffnen, Passwort eingeben. Nach 5 Fehlversuchen 15 Minuten Sperre.
Ausschalten: `python run.py fernzugriff aus`.

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

## Diagnose: Warum handelt der Bot nicht?

```
python run.py diagnose            # letzte 90 Tage (dauert einige Minuten)
```
Zeigt fuer die letzten 14 Tage, wie viele Signale die Strategie hatte und welcher Filter wie viele
davon aussortiert hat. Danach wird jeder Filter einzeln abgeschaltet und ehrlich verglichen
(Lernen mit den ersten 2/3, Pruefen am letzten Drittel): mehr Trades - aber auch noch profitabel?
Am Ende steht eine Empfehlung. Ergebnis auch in `data/diagnose.csv`.

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
