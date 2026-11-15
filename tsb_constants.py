"""Konstanten: Regler-Grenzen, feste Annahmen, Beschriftungen, Farben und Presets."""

# ---------------------------------------------------------------- Netz
L_MIN, L_MAX, DEFAULT_L = 2, 6, 4            # Teilstrecken (Zug) bzw. Nächte (Hotel)
CAP_MIN, CAP_MAX, DEFAULT_CAP = 2, 16, 8     # Plätze bzw. Zimmer je Teilstrecke
STATES_LIVE_MAX = 120_000                    # so viele Zustände (Kapazität + 1) ** L darf die exakte DP live höchstens haben (rund 110 MB)
RHO_STEPS = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)    # Last: erwartete Nachfrage je Teilstrecke geteilt durch die Kapazität
DEFAULT_RHO = 2.0
ALPHA_STEPS = (0.2, 0.3, 0.4, 0.6, 0.8, 1.0)           # Preisgefälle: Preis = 10 · Länge ** alpha; 1 = gleicher Preis je Teilstrecke
DEFAULT_ALPHA = 0.3
EARLY_STEPS = (0.0, 0.5, 1.0)                           # Buchungszeit: 0 = alle Längen gleichmäßig, 1 = lange Fahrten früh, kurze spät
DEFAULT_EARLY = 1.0
ERR_STEPS = (-0.5, -0.25, 0.0, 0.25, 0.5)               # Verzerrung der Prognose für lange Fahrten (Faktor 1 + err)
DEFAULT_ERR = 0.0

KULISSEN = ("zug", "hotel")
DEFAULT_KULISSE = "zug"
GAMMA = {"zug": 0.5, "hotel": 0.8}           # Nachfrage der Länge s ∝ exp(-GAMMA · (s - 1)): im Hotel sind kurze Aufenthalte häufiger
FARE_BASE = 10.0
LOAD_PER_PERIOD = 0.5                        # erwartete Anfragen je Periode (Zahl der Perioden T = Nachfrage / 0,5)
MAX_ARRIVAL = 0.98                           # Summe der Ankunftswahrscheinlichkeiten je Periode höchstens so groß

RULES = ("fcfs", "bid_static", "bid_resolve", "dp")
RULE_LABELS = {
    "fcfs": "Annahme nach Eingang",
    "bid_static": "Bid-Price, einmal gerechnet",
    "bid_resolve": "Bid-Price, je Periode neu",
    "dp": "Exakte Rechnung (DP)",
}
RULE_SHORT = {"fcfs": "Eingang", "bid_static": "Bid-Price", "bid_resolve": "Bid-Price neu", "dp": "DP"}
DEFAULT_RULE = "fcfs"

KULISSE_LABELS = {"zug": "Zug mit Halten", "hotel": "Hotel mit Mehrnacht-Aufenthalt"}
WORDS = {
    "zug": {"segment": "Teilstrecke", "segments": "Teilstrecken", "trip": "Fahrt", "trips": "Fahrten", "unit": "Platz", "units": "Plätze", "short": "kurze Fahrten", "long": "lange Fahrten", "long_dat": "langen Fahrten"},
    "hotel": {"segment": "Nacht", "segments": "Nächte", "trip": "Aufenthalt", "trips": "Aufenthalte", "unit": "Zimmer", "units": "Zimmer", "short": "kurze Aufenthalte", "long": "lange Aufenthalte", "long_dat": "langen Aufenthalten"},
}

# ---------------------------------------------------------------- Zufall
SEED_MAX, DEFAULT_SEED = 999999, 35
LIVE_SEQ = 200                               # Anfragefolgen der Live-Rechnung
SUBSEED_STRIDE = 1000                        # Zufalls-Seed der Folge k: seed · 1000 + k

# ---------------------------------------------------------------- Farben (Fahrtlänge 1 bis 6)
LENGTH_COLORS = ("#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b")
COLOR_LIMIT = "#c0392b"
COLOR_BARS = {"fcfs": "#d98e3f", "bid_static": "#4c78a8", "bid_resolve": "#2F6B65", "dp_hat": "#8c6bb1", "dp": "#555555"}

# ---------------------------------------------------------------- Presets (Felder von Params)
PRESETS = {
    "Billige lange Fahrten, Überlast": {},   # der Standard: alle Regler auf Voreinstellung
    "Gleicher Preis je Teilstrecke": {"alpha": 1.0, "early": 0.0},
    "Mäßige Last, kleiner Zug": {"cap": 4, "rho": 1.2, "alpha": 0.6, "early": 0.0},
    "Fast leerer Zug": {"rho": 0.8, "alpha": 0.6, "early": 0.0},
    "Starke Überlast, gleichmäßig gebucht": {"rho": 3.0, "alpha": 0.6, "early": 0.0},
    "Falsche Prognose (+50 %)": {"err": 0.5},
    "Großes Netz (6 Teilstrecken)": {"L": 6, "cap": 6, "rho": 1.5},
    "Hotel mit Mehrnacht-Aufenthalt": {"kulisse": "hotel", "L": 5, "alpha": 0.6},
}
PRESET_HELP = {
    "Billige lange Fahrten, Überlast": "4 Teilstrecken mit je 8 Plätzen, doppelt so viel Nachfrage wie Plätze, lange Fahrten je Teilstrecke deutlich billiger und früh gebucht: Annahme nach Eingang verdrängt die teuren kurzen Fahrten.",
    "Gleicher Preis je Teilstrecke": "Preis proportional zur Länge: Ein einmal gerechneter Bid-Price nimmt dieselben Fahrten an wie die Annahme nach Eingang; erst Regeln, die den Zustand der Belegung berücksichtigen (Neuberechnung, exakte Rechnung), holen etwas von der Lücke zurück.",
    "Mäßige Last, kleiner Zug": "4 Plätze je Teilstrecke, Last 1,2: der einmal gerechnete Bid-Price lehnt zu viel ab und ist schlechter als die Annahme nach Eingang.",
    "Fast leerer Zug": "Last 0,8: kaum Konkurrenz um die Plätze, alle Regeln liegen dicht beieinander.",
    "Starke Überlast, gleichmäßig gebucht": "Dreifache Nachfrage, alle Fahrtlängen gleichmäßig über die Buchungszeit verteilt: auch ohne frühe Buchung der langen Fahrten verdrängen sie die kurzen.",
    "Falsche Prognose (+50 %)": "Die Regeln rechnen mit 50 % mehr Nachfrage nach langen Fahrten, als tatsächlich kommt; die Referenz kennt die wahre Nachfrage.",
    "Großes Netz (6 Teilstrecken)": "6 Teilstrecken mit je 6 Plätzen, Last 1,5: 117.649 Zustände, so viele rechnet die exakte Rechnung live gerade noch.",
    "Hotel mit Mehrnacht-Aufenthalt": "5 Nächte mit je 8 Zimmern, kurze Aufenthalte häufiger, Mehrnacht-Aufenthalte je Nacht billiger: dasselbe Modell wie der Zug.",
}
