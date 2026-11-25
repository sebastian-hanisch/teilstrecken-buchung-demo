# Zug mit Halten: kurze oder lange Fahrt annehmen? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-teilstrecken-buchung-demo.streamlit.app/)**

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Themenseite „Sitzplätze optimieren“). Ein **Zug mit mehreren Halten** (oder ein **Hotel mit Mehrnacht-Aufenthalten**): Eine Buchung belegt je einen Platz auf mehreren **Teilstrecken** zugleich, und jede Anfrage muss sofort angenommen oder abgelehnt werden. Die Demo vergleicht vier Annahmeregeln, alle gemessen an der **exakten Rechnung** (Dynamische Programmierung über alle Belegungen): **Annahme nach Eingang**, **Bid-Price einmal gerechnet** (Schattenpreis je Teilstrecke aus einem Optimierungsproblem), **Bid-Price je Periode neu** und die exakte Rechnung selbst.

## Warum dieses Problem

Wer einen Platz für die lange Fahrt vergibt, verbaut ihn für mehrere kurze (bei zwei Teilstrecken für zwei). Die Demo fragt: **Was kostet es, Buchungen in der Reihenfolge ihres Eingangs anzunehmen, wenn lange Fahrten je Teilstrecke billiger sind, welche Regel holt den Verlust zurück, und wie genau muss die Prognose der Nachfrage dafür sein?** Der Zuschnitt (mehrere Teilstrecken je Buchung, ein Zustandsvektor statt einer einzelnen Kapazität) unterscheidet die Demo von `revenue-management-demo` (eine Ressource, zwei Klassen) und `uld-buchungsannahme-demo` (ein Gewichtslimit). Das Verfahren selbst (Bid-Price in Netzen, Talluri und van Ryzin 1998) ist Lehrbuchstoff; neu ist der Zuschnitt und das Messen der Regeln an der exakten Rechnung.

## Modell

Eine Linie mit L Teilstrecken (Zug: 2 bis 6; Hotel: Nächte), auf jeder K Plätze (2 bis 16, je nach L begrenzt, damit die exakte Rechnung live bleibt). Ein **Produkt** (i, j) belegt je einen Platz auf den Teilstrecken i + 1 bis j und kostet 10 · s^α mit s = j − i; α = 1 heißt gleicher Preis je Teilstrecke, kleinere α heißen: lange Fahrten sind je Teilstrecke billiger. In jeder von T Perioden kommt höchstens eine Anfrage, Produkt p mit fester Wahrscheinlichkeit; die Nachfrage je Produkt der Länge s ist proportional zu e^(−γ (s−1)) (γ = 0,5 im Zug, 0,8 im Hotel). Die Zahl der Perioden ist T = ⌈Σ D_p / 0,5⌉ mit der erwarteten Gesamtnachfrage D_p je Produkt, und die Summe der Ankunftswahrscheinlichkeiten je Periode ist höchstens 0,98. Die **Last** ρ ist die mittlere erwartete Nachfrage je Teilstrecke (Mittel über die Teilstrecken) geteilt durch K; die **Buchungszeit** legt fest, ob alle Längen gleichmäßig über die Perioden gebucht werden (0) oder lange Fahrten früh und kurze spät (1; Zwischenwert 0,5). Der zeitliche Verlauf eines Produkts ist 1 + e σ_p (1 − 2τ) mit τ = (t − ½)/T, σ_p = (s_p − s̄)/(s_max − s̄) (s̄: Mittel der Längen aller Produkte) und der Buchungszeit e, nach unten bei 0,05 gekappt und je Produkt auf Mittel 1 normiert. Eine Regel entscheidet sofort; „passt nicht“ heißt immer „abgelehnt“. Alle Formeln stehen im 📐-Expander der App (vollständig: Variablen, Zustandsraum, Rekursion, Optimierungsproblem und seine Duale, Prognose).

**Ertrag in % der exakten Rechnung** = mittlerer Ertrag der Regel geteilt durch den mittleren Ertrag der exakten Rechnung auf denselben Anfragefolgen mal 100 (gepaarte Vergleiche). Der **Standardfehler** gehört zur gepaarten Differenz.

## Methodik

- **Annahme nach Eingang (First-Come-First-Served, FCFS):** nimmt jede Anfrage an, die passt.
- **Bid-Price einmal gerechnet:** nimmt an, wenn der Preis mindestens die Summe der Schattenpreise der belegten Teilstrecken deckt. Die Schattenpreise sind die Duale der Kapazitätsbedingungen von „größter Ertrag bei erwarteter Nachfrage und voller Kapazität“ (`scipy.optimize.linprog`, HiGHS); bei entartetem Problem sind die Duale nicht eindeutig, die Demo nimmt die, die der Löser liefert, und die Aussagen über Bid-Price gelten für diese Wahl. Bei der Neuberechnung zählt die Restnachfrage die laufende Periode mit.
- **Bid-Price je Periode neu:** dasselbe, aber das Problem wird in jeder Periode mit den freien Plätzen und der Restnachfrage neu gelöst.
- **Exakte Rechnung:** Rückwärtsrekursion über alle (K + 1)^L Belegungen; angenommen wird, wenn Preis plus Wert der Belegung danach mindestens den Wert der jetzigen Belegung erreicht. Der erwartete Ertrag ist ein exakter Wert (kein Simulationsergebnis).
- **Fester Schutz (Protection Level):** eine lange Fahrt wird nur angenommen, wenn danach auf jeder ihrer Teilstrecken noch k Plätze frei bleiben; alles andere nach Eingang.
- **Prognose-Szenarien (4 Teilstrecken, 8 Plätze):** A Last 2, α 0,3, früh; B Last 1,2, α 0,6, gleichmäßig; C Last 3, α 0,6, früh; D Last 1,2, α 1, gleichmäßig; die fünf Verzerrungsstufen sind −50, −25, 0, +25 und +50 %. Das Hotel-Raster hat 9 Zellen (5 Nächte, 8 Zimmer, Last 1,2 / 2 / 3, α 1 / 0,6 / 0,3, lange Aufenthalte früh).
- **Prognose:** Die Regeln rechnen mit einer Nachfrage, in der die Nachfrage nach langen Fahrten (mehr als die halbe Linie) mit 1 + Fehler multipliziert ist (−50 bis +50 %); gespielt werden die wahren Folgen. Die Referenz kennt die wahre Nachfrage.
- **Live:** 200 Anfragefolgen mit Zufalls-Seeds `Zufalls-Seed · 1000 + k` aus einem eigenen SplitMix64; dazu der **exakte Erwartungswert** von Annahme nach Eingang, einmal gerechnetem Bid-Price und der exakten Rechnung mit Prognose (Zustandsrechnung ohne Simulation). Für die Neuberechnung gibt es keinen exakten Wert, nur die Simulation.
- **Vorgerechnet:** `tools/make_results.py` erzeugt `data/tsb_results.json` (Raster und Hotel je Zelle 1.000 Folgen, Prognose und Netzgröße je Zeile 500 Folgen mit Neuberechnung, Schutz: k auf 500 Trainingsfolgen gewählt, auf 1.000 frischen Folgen gemessen). Jede Zelle hat einen eigenen Zufalls-Seed.

## Befunde (gemessen, keine Behauptungen)

Alle Prozentzahlen sind Ertrag in % der exakten Rechnung mit wahrer Nachfrage; „Raster“ heißt die Messung `raster` (Zug mit 4 Teilstrecken, 72 Zellen aus 4, 8 und 16 Plätzen je Teilstrecke, Last 0,8 / 1,2 / 2 / 3, α 1 / 0,6 / 0,3, Buchungszeit gleichmäßig und früh, je 1.000 Folgen). Nachfrage und Preise sind gesetzt, nicht an Buchungsdaten kalibriert: Die Zahlen gelten für dieses Modell.

| Frage | Befund | Beleg |
|---|---|---|
| Was kostet die Annahme nach Eingang? | Im Raster erreicht sie **74,2 bis 99,9 %**; bei Last 0,8 mindestens **98,7 %**; in den **24 Zellen** mit Last ab 2 und α höchstens 0,6 zwischen **74,2 und 96,7 %**. Der Standardfehler der Lücke ist in jeder Zelle höchstens 0,34 Prozentpunkte. | `test_claims.py::test_raster_fcfs` |
| Voreinstellung (4 Teilstrecken, 8 Plätze, Last 2, α 0,3, früh, Zufalls-Seed 35) | Exakter Erwartungswert: Annahme nach Eingang **86,3 %**, Bid-Price einmal gerechnet **97,9 %** (der exakte Ertrag der exakten Rechnung ist 257,7). Simulation mit 200 Folgen: 86,6 / 97,9 / 99,1 % (Annahme nach Eingang / Bid-Price einmal / je Periode neu; Standardfehler 0,5 / 0,2 / 0,2 Prozentpunkte). Von den angefragten Fahrten der Länge 1 nimmt die Annahme nach Eingang (Simulation mit 200 Folgen; lange Fahrten sind hier die mit 3 und 4 Teilstrecken) 54 % an, von den langen 49 %; die exakte Rechnung 85 % und 1 %. | `test_claims.py::test_default_numbers` |
| Hilft der **einmal gerechnete** Bid-Price bei gleichem Preis je Teilstrecke (α = 1)? | **Nein:** In allen **24 Zellen** mit α = 1 nimmt er in jeder der 1.000 Folgen genau dieselben Anfragen an wie die Annahme nach Eingang; deren Lücke zur exakten Rechnung bleibt (**95,7 bis 99,9 %**). Dasselbe gilt in 12 weiteren Zellen (Last 0,8, α 0,6 und 0,3): zusammen in **36 von 72 Zellen** gleiche Entscheidungen. Die Neuberechnung je Periode ist im Raster nicht gemessen; im Prognose-Szenario D (α = 1, Last 1,2, 8 Plätze, je Verzerrungsstufe 500 Folgen) erreicht sie **98,1 bis 99,0 %** gegen **96,6 bis 96,8 %** für die Annahme nach Eingang. | `test_claims.py::test_alpha_one_is_identical` |
| Wann ist der einmal gerechnete Bid-Price **schlechter** als die Annahme nach Eingang? | In **7 von 72 Zellen** (Unterschied mehr als zwei Standardfehler der gepaarten Differenz unter null; bei 72 Zellen sind rund 3 Treffer allein durch Zufall zu erwarten, der exakte Erwartungswert unten ist frei von Simulationsrauschen), alle bei Last 1,2 und α höchstens 0,6: bei 4 Plätzen je Teilstrecke −1,0 bis −2,0, bei 8 Plätzen −0,6 bis −0,8 Prozentpunkte; keine Zelle mit 16 Plätzen. In **28 Zellen** ist er um mehr als zwei Standardfehler besser (bis +25,0 Prozentpunkte). Nach dem exakten Erwartungswert beider Regeln (ohne Simulation) ist er in **8 Zellen** schlechter, die achte (8 Plätze, Last 1,2, α 0,3, früh) mit −0,2 Prozentpunkten; auch diese 8 liegen alle bei Last 1,2, nie bei 16 Plätzen. | `test_claims.py::test_bid_price_worse_than_first_come` |
| Wie weit kommt der einmal gerechnete Bid-Price? | Im Raster **95,6 bis 99,9 %**; in den 24 Zellen mit Last ab 2 und α höchstens 0,6 zwischen **95,6 und 99,2 %**. | `test_claims.py::test_raster_bid_price` |
| Schließt die Neuberechnung die Lücke? | In der Prognose-Messreihe (Netz mit 4 Teilstrecken und 8 Plätzen, 4 Szenarien × 5 Verzerrungen, je 500 Folgen) erreicht der Bid-Price je Periode neu **98,1 bis 99,9 %** (einmal gerechnet: 96,2 bis 98,6 %); in der Netzgrößen-Messreihe 98,7 bis 99,6 %. | `test_claims.py::test_neuberechnung` |
| Wie genau muss die Prognose sein? | Die **exakte Rechnung mit verzerrter Prognose** (Nachfrage nach langen Fahrten um −50 bis +50 % falsch) verliert nach dem exakten Erwartungswert in allen 20 Zeilen **höchstens 0,42 %** (des Ertrags der exakten Rechnung); in den Szenarien A und C (Überlast, lange Fahrten billig) weniger als 0,05 %. Gemessen nur für diese Verzerrung in diesen vier Szenarien. | `test_claims.py::test_forecast` |
| Wächst die Lücke mit der Größe des Netzes? | In dieser Messreihe (2 bis 6 Teilstrecken, 6 Plätze, Last 2, α 0,3, früh, je 500 Folgen; nur diese eine Kapazität) bei der Annahme nach Eingang nicht: **85,6 bis 88,0 %** bei jeder Größe. Der einmal gerechnete Bid-Price steigt nicht gleichmäßig (99,9 / 94,0 / 97,7 / 98,1 / 96,2 %), liegt aber ab 3 Teilstrecken mit 94,0 bis 98,1 % unter dem Wert bei 2 Teilstrecken (99,9 %). | `test_claims.py::test_net_size` |
| Reicht ein fester Schutz für lange Fahrten? | Nein: Mit dem auf Trainingsfolgen aus k = 0 bis 8 gewählten k (k = 8 heißt: keine lange Fahrt annehmen) erreicht er in Szenario A 96,3 % (k = 8; Annahme nach Eingang 86,2 %), in C 94,0 % (k = 8; 86,1 %), der einmal gerechnete Bid-Price 98,2 und 97,2 %. In B und D ist k = 0 am besten: kein Schutz hilft. | `test_claims.py::test_protection` |
| Hotel? | Mit 5 Nächten, 8 Zimmern, früh gebuchten langen Aufenthalten: bei α = 1 entscheidet der Bid-Price in allen drei Zellen wie die Annahme nach Eingang; bei Last 3 und α 0,3 erreicht die Annahme nach Eingang **77,5 %**, der Bid-Price **99,7 %**. | `test_claims.py::test_hotel` |
| Stimmt die Simulation? | In **3 von 72** Rasterzellen weicht der simulierte Ertrag der exakten Rechnung um mehr als zwei Standardfehler vom exakten Erwartungswert ab; dort sind es höchstens 0,07 %, weil der Standardfehler fast null ist. Die größte Abweichung über alle Zellen ist **0,81 %** und liegt innerhalb von zwei Standardfehlern. Alle Zellen nutzen verschiedene Zufalls-Seeds. | `test_claims.py::test_simulation_check` |

**Messung dieser README, nicht im Test:** die Rechenzeit (die exakte Rechnung braucht auf dem Entwicklungsrechner bei 117.649 Zuständen rund 0,3 Sekunden, bei 531.441 rund 7 Sekunden und 455 MB; Zeiten hängen vom Rechner ab).

### Befunde und Korrekturen gegenüber der Vormessung

Die Vormessung (andere Zufallsfolgen, 200 bis 300 Folgen je Zelle, noch ohne Standardfehler) gab −25,9 % für die Annahme nach Eingang und −5,2 % für den einmal gerechneten Bid-Price als größte Abstände und acht Zellen mit schlechterem Bid-Price; mit 1.000 Folgen und den Zufallsfolgen der Demo sind es −25,8 %, −4,4 % und sieben Zellen. Die Aussage der Vormessung, die Prognose müsse „kaum“ stimmen, ist jetzt auf das Gemessene begrenzt (höchstens 0,42 % in vier Szenarien).

## Ehrliche Grenzen

- **Nachfrage, Preise und Buchungszeit sind gesetzt** (Last, α, Buchungszeit, Nachfrage fällt mit der Länge exponentiell), nicht an Buchungsdaten kalibriert. Die Prozentzahlen sind Ergebnisse dieses Modells, keine Prognose für Bahnen oder Hotels.
- **Ein Preis je Produkt**, keine Preisklassen, keine Stornierungen und No-Shows, keine Überbuchung, kein Umbuchen, **keine Sitzplatzzuweisung** (nur die Kapazität je Teilstrecke wird geprüft).
- Die **exakte Rechnung** ist ein Maßstab für den besten Ertrag einer sofort entscheidenden Regel; sie rechnet nur bei kleinen Netzen (live höchstens 120.000 Zustände). Aussagen über Bid-Price gelten für die einfachste Umsetzung (Schattenpreise aus dem Optimierungsproblem), nicht für jede Bid-Price-Variante.
- **Prognosefehler:** nur die Nachfrage nach langen Fahrten, um höchstens 50 %, in vier Szenarien. Fehler bei den Preisen, den kurzen Fahrten oder im zeitlichen Verlauf sind nicht gemessen. Die Neuberechnung und die exakte Rechnung mit Prognose zeigen die Wirkung einer Verzerrung, nicht die Qualität eines geschätzten Modells (`revenue-management-demo` nennt die Prognoseunschärfe als natürlichen Baustein für einen ML-Teil; hier ist sie nur ein Regler).
- Die Neuberechnung hat keinen exakten Erwartungswert; ihre Zahlen sind Simulationen mit Standardfehler.
- Eine Linie ohne Verzweigungen.

## Verifikation

- **Kern:** Handrechnung an zwei Mini-Instanzen mit zwei Teilstrecken und einem Platz (Werte 13,45 / 11,5 / 12,5 / 15 von Hand) und Zulässigkeit (nie mehr Belegung als Kapazität) für alle Regeln auf mehreren Netzen (`tests/test_algorithm.py`).
- **Orakel (`tests/test_oracle_teilstrecken_buchung.py`):** die exakte Rechnung gegen eine Baumrekursion in Brüchen und gegen die Aufzählung **aller** zustandsabhängigen Regeln an drei Mini-Instanzen (zwei Teilstrecken, ein Platz, drei Perioden; bewertet durch Aufzählung aller Anfragefolgen); der exakte Erwartungswert der Regeln gegen eine Rekursion mit Wörterbuch; Simulation gegen diesen exakten Wert (Abstand höchstens 4,5 Standardfehler); Bid-Price-Schattenpreise über die starke Dualität (zweites LP); Annahme nach Eingang gegen eine zweite Umsetzung.
- **Verdikte:** jede Schwelle und jede Zahl im Text an handgesetzten Werten (`tests/test_verdicts.py`); Zusammenfassungen der Messreihen an handgemachten Zellen.
- **Presets:** vollständig, Werte in den Reglergrenzen, Standard-Preset gleich Standardeinstellung, paarweise verschieden (`tests/test_presets.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button; die beiden Schieberegler und die vier Auswahlschieberegler am Minimum und Maximum bzw. an der ersten und letzten Auswahl) und Footer-Test; Layout-Test (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Regleränderung hebt die Markierung auf, die Kapazität folgt der Grenze der Live-Rechnung).
- **Zahlen der README:** `tests/test_claims.py` rechnet jede Zahl der Befundtabelle (außer den Zahlen der Vormessung und der Rechenzeit) aus `data/tsb_results.json` und aus der Live-Analyse nach und prüft, dass sie in dieser README steht; dazu die Zählung der Messreihen und die Reproduzierbarkeit einer Zelle (dieselben Zufalls-Seeds, dieselben Werte).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `data/tsb_results.json` | vorgerechnete Messreihen (Raster, Hotel, Prognose, Netzgröße, Schutz, Zustände) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/make_results.py` | erzeugt `data/tsb_results.json` (`python tools/make_results.py raster hotel prognose netz schutz zustaende`) |
| `tsb_algorithm.py` | Kern: exakte Rechnung, Bid-Price (LP-Duale), vier Annahmeregeln, exakter Erwartungswert einer Regel |
| `tsb_constants.py` | Regler-Grenzen, feste Annahmen, Beschriftungen, Farben, Presets |
| `tsb_evaluation.py` | Live-Analyse, Verdikte, Zusammenfassungen der Messreihen |
| `tsb_layout.py` | Fester Kopf (Stylesheet) |
| `tsb_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `tsb_rng.py` | eigener Zufallsgenerator (SplitMix64) |
| `tsb_scenario.py` | Netz, Produkte, Preise, Ankunftswahrscheinlichkeiten, Folgenerzeuger |
| `tsb_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Handrechnung, Orakel, Verdikte, Randfälle, Presets, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe, Zahlen der README |

## Bewusst nicht umgesetzt

- **Stornierungen, No-Shows und Überbuchung**, mehrere Preisklassen je Produkt, Umbuchen: eigene Modelle (siehe die READMEs der Nachbar-Demos).
- **Sitzplatzzuweisung** (welcher Sitz?): hier zählt nur die Kapazität je Teilstrecke.
- **Geschätzte statt verzerrte Prognose** (ein Lernverfahren für die Nachfrage): der Regler zeigt die Wirkung eines Fehlers, nicht die eines Schätzers.
- **Netze mit Verzweigungen** und große Netze, für die die exakte Rechnung nicht mehr rechnet (statt ihrer Näherungen wie die Bid-Price-Variante mit Neuberechnung nur als Vergleich).
- **Kalibrierung** der Nachfrage und der Preise an Buchungsdaten.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zum Thema: [Sitzplätze optimieren](https://sebastianhanisch.net/sitzplaetze-optimieren.html).
