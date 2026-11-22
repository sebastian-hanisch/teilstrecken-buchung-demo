"""Zug mit Halten: kurze oder lange Fahrt annehmen? - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Zug mit mehreren Teilstrecken (oder ein Hotel mit Mehrnacht-Aufenthalten): Jede Buchung belegt mehrere Teilstrecken zugleich, und jede Anfrage muss sofort angenommen oder abgelehnt werden.
Die Demo vergleicht vier Annahmeregeln mit der exakten Rechnung (Dynamische Programmierung über alle Belegungen): Annahme nach Eingang, Bid-Price einmal gerechnet, Bid-Price je Periode neu und die exakte Rechnung selbst.

Lauffähig mit: streamlit run app.py
"""

import time

import pandas  # noqa: F401  (früh importiert, Vorsichtsmaßnahme gegen einen Importfehler beim Kaltstart der Cloud-App)
import streamlit as st

import tsb_algorithm as A
import tsb_constants as C
from tsb_evaluation import (
    Settings,
    analyse,
    forecast_summary,
    alpha_one_resolve,
    load_results,
    raster_summary,
    verdict_forecast,
    verdict_fcfs,
    verdict_rules,
    words,
)
from tsb_layout import dunkles_design, kopf_css
from tsb_presets import (
    bounds,
    clamp_capacity,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from tsb_scenario import Params, build_net, cap_max, n_states
from tsb_visualization import build_occupancy, build_rules

st.set_page_config(page_title="Zug mit Halten – Sebastian Hanisch", layout="wide")

DATA = load_results()


@st.cache_data(show_spinner=False)
def _analysis(settings):
    t = time.perf_counter()
    result = analyse(settings)
    return result, time.perf_counter() - t       # die Rechenzeit der ersten Rechnung bleibt beim Zwischenspeichern erhalten


def _show(verdict):
    kind, text = verdict
    {"success": st.success, "warning": st.warning, "info": st.info}[kind](text)


def _num(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def _sgn(x, digits=1):
    return f"{x:+.{digits}f}".replace(".", ",").replace("-", "−")


def _thousands(n):
    return f"{n:,}".replace(",", ".")


def _pct(x):
    return "—" if x is None else f"{100 * x:.0f} %"


EARLY_LABELS = {0.0: "gleichmäßig", 0.5: "etwas früher", 1.0: "früh"}

ASSUMPTIONS_MARKDOWN = """
| Annahme | Wer setzt an |
|---|---|
| Jede Buchung belegt je einen Platz auf allen Teilstrecken ihrer Fahrt; geprüft wird nur die Kapazität je Teilstrecke, nicht, welcher Sitz es ist | Modell |
| In jeder Periode kommt höchstens eine Anfrage, das Produkt nach festen Wahrscheinlichkeiten; kurze Fahrten sind häufiger (Zug: Faktor e^-0,5 je Teilstrecke, Hotel: e^-0,8 je Nacht) | Modell |
| Der Preis einer Fahrt der Länge s ist 10 · s^α; α = 1 heißt gleicher Preis je Teilstrecke, kleinere α heißen: lange Fahrten sind je Teilstrecke billiger | Regler, Modell |
| Die Buchungszeit (gleichmäßig oder lange Fahrten früh) und die Last sind gesetzt, nicht an Buchungsdaten kalibriert | Regler |
| Die Regeln kennen die Nachfrage als Prognose; der Regler „Prognose für lange Fahrten“ verzerrt nur die Nachfrage nach langen Fahrten (mehr als die halbe Linie) | Regler |
| Ein Preis je Produkt, keine Fahrpreisklassen, keine Stornierungen und No-Shows, keine Überbuchung, kein Umbuchen, keine Sitzplatzzuweisung | Nicht umgesetzt |
| Die exakte Rechnung ist der Maßstab für den bestmöglichen erwarteten Ertrag einer Annahmeregel, kein Verfahren, das ein Betreiber so einsetzen könnte, wenn das Netz groß ist | Modell |
"""

# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in tsb_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🚆 Zug mit Halten: kurze oder lange Fahrt annehmen?")
st.markdown(
    """
In einem **Zug mit mehreren Halten** (oder einem **Hotel mit Mehrnacht-Aufenthalten**) belegt eine Buchung mehrere **Teilstrecken** zugleich, und jede Anfrage muss sofort angenommen oder abgelehnt werden.
Die Demo fragt: **Was kostet es, Buchungen in der Reihenfolge ihres Eingangs anzunehmen, welche Regel holt den Verlust zurück, und wie genau muss die Prognose der Nachfrage sein?**
Verglichen werden die **Annahme nach Eingang**, ein **Bid-Price** (Schattenpreis je Teilstrecke), einmal gerechnet und je Periode neu, jeweils an der **exakten Rechnung** (Dynamische Programmierung über alle Belegungen) gemessen.

Nachfrage und Preise sind Zufallsmodelle, keine Buchungsdaten. Weiter unten: die **Belegung** eines Beispielzuges, drei **Kernfragen**, sechs **Experimente** mit vorgerechneten Messreihen und die Expander
**🔧 Wie wir das erreichen – vollständiger Methodenvergleich**, **📐 Mathematische Formulierung** und **🚧 Wo die Annahmen enden**. Wer es ohne Fachwörter lesen will, öffnet **💬 In einfachen Worten**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/teilstrecken-buchung-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein Zug mit mehreren Halten, oder ein Hotel mit Aufenthalten über mehrere Nächte: Jede Buchung belegt einen Platz auf mehreren Teilstrecken zugleich, und jede Anfrage muss sofort angenommen oder abgelehnt werden. Eine lange Fahrt kann so Plätze belegen, die mehrere kurze, besser bezahlte Fahrten gebraucht hätten.

Die App vergleicht die Annahme nach Eingang mit zwei Regeln, die für jede Teilstrecke einen Schattenpreis rechnen, und mit der bestmöglichen Regel, die sie exakt ausrechnet. Nachfrage und Preise sind ausgedacht, nicht aus Buchungsdaten.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/teilstrecken-buchung-erklaert.html)
        """
    )
# erklaerseite:einfach:end
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
clamp_capacity()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🚆 Netz")
    kulisse = st.selectbox("Kulisse", list(C.KULISSEN), key="kulisse_select", format_func=lambda k: C.KULISSE_LABELS[k],
                           help="Dasselbe Modell, andere Wörter: Im Hotel sind kurze Aufenthalte häufiger als im Zug kurze Fahrten (Nachfrage fällt mit der Länge schneller).")
    w = C.WORDS[kulisse]
    L = st.slider(w["segments"], *bounds("L_slider"), key="L_slider", help="Länge der Linie. Die exakte Rechnung braucht (Kapazität + 1) hoch Zahl der Teilstrecken Zustände.")
    clamp_capacity()
    cap = st.slider(f"{w['units']} je {w['segment']}", C.CAP_MIN, cap_max(L), key="cap_slider",
                    help=f"Obergrenze {cap_max(L)}: so viele {w['units']} rechnet die exakte Rechnung bei {L} {w['segments']} noch live (höchstens {_thousands(C.STATES_LIVE_MAX)} Zustände).")
    st.header("💶 Nachfrage und Preise")
    rho = st.select_slider("Last", options=list(C.RHO_STEPS), key="rho_select", format_func=lambda v: _num(v, 1),
                           help=f"Erwartete Nachfrage je {w['segment']} geteilt durch die Kapazität. Unter 1 ist genug Platz, über 1 konkurrieren die Buchungen.")
    alpha = st.select_slider("Preisgefälle α", options=list(C.ALPHA_STEPS), key="alpha_select", format_func=lambda v: _num(v, 1),
                             help=f"Preis = 10 · Länge hoch α. 1 = gleicher Preis je {w['segment']}; kleiner heißt: {w['long']} sind je {w['segment']} billiger.")
    early = st.select_slider("Buchungszeit", options=list(C.EARLY_STEPS), key="early_select", format_func=lambda v: EARLY_LABELS[v],
                             help=f"„gleichmäßig“: alle Längen über die Buchungszeit gleich verteilt; „früh“: {w['long']} werden früh gebucht, kurze spät.")
    st.header("🔮 Prognose")
    err = st.select_slider(f"Prognose für {w['long']}", options=list(C.ERR_STEPS), key="err_select", format_func=lambda v: "0 %" if v == 0 else f"{100 * v:+.0f} %".replace("-", "−"),
                           help=f"Die Regeln rechnen mit der wahren Nachfrage nach {w['long_dat']} mal (1 + Wert). Die Referenz kennt die wahre Nachfrage. Die Annahme nach Eingang braucht keine Prognose.")
    st.header("👁️ Ansicht")
    rule = st.selectbox("Regel im Beispiel", list(C.RULES), key="rule_select", format_func=lambda k: C.RULE_LABELS[k],
                        help="Welche Regel das Beispiel (die erste Anfragefolge) zeigt. Die Kennzahlen und Kernfragen zeigen immer alle Regeln.")
    st.header("🎲 Zufall")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", help="Gleicher Zufalls-Seed, gleiche Anfragefolgen (eigener Zufallsgenerator).")
    st.button("🎲 Neuen Zufalls-Seed würfeln", on_click=randomize_seed)

params = Params(kulisse=kulisse, L=L, cap=cap, rho=float(rho), alpha=float(alpha), early=float(early), err=float(err))
settings = Settings(params=params, seed=int(seed))
sync_query_params({"kulisse_select": kulisse, "L_slider": L, "cap_slider": cap, "rho_select": float(rho), "alpha_select": float(alpha), "early_select": float(early), "err_select": float(err),
                   "rule_select": rule, "seed_input": int(seed)})

with st.spinner(f"Die {C.LIVE_SEQ} Anfragefolgen werden gerechnet: vier Regeln und die exakte Rechnung …"):
    a, elapsed = _analysis(settings)
ex = a.examples[rule]

# ---------------------------------------------------------------------------------------------- Ergebnis
st.markdown("## Ergebnis des eingestellten Netzes")
c1, c2, c3, c4 = st.columns(4)
for col, key in zip((c1, c2, c3), ("fcfs", "bid_static", "bid_resolve")):
    s = a.rules[key]
    col.metric(C.RULE_LABELS[key], f"{_num(100 + s.gap.mean)} %", delta=f"± {_num(s.gap.se)} Standardfehler", delta_color="off", help="Mittlerer Ertrag in Prozent der exakten Rechnung (= 100 %), gepaarte Folgen.")
c4.metric("Exakte Rechnung", f"{_num(a.v0, 1)}", delta="erwarteter Ertrag = 100 %", delta_color="off", help="Exakter Erwartungswert des Ertrags bei wahrer Nachfrage (Rückwärtsrekursion über alle Belegungen).")
st.caption(f"Rechenzeit der Live-Rechnung: {elapsed:.1f} s ({a.n} Anfragefolgen mit Zufalls-Seeds {int(seed)} · 1000 + k, je {a.T} Perioden; {_thousands(a.states)} Zustände; wiederholte Einstellungen kommen aus dem Zwischenspeicher). "
           f"Kontrolle: Der simulierte Ertrag der exakten Rechnung weicht um {_sgn(a.sim_vs_v0.mean)} % (Standardfehler {_num(a.sim_vs_v0.se)}) vom exakten Erwartungswert ab.")
st.plotly_chart(build_rules(a, params), width="stretch", config={"displayModeBar": False}, key="plot_rules")
st.markdown(f"**Beispiel: die erste Anfragefolge unter „{C.RULE_LABELS[rule]}“** (Belegung am Ende, gestapelt nach Länge der Fahrt)")
st.plotly_chart(build_occupancy(ex, params, None), width="stretch", config={"displayModeBar": False}, key="plot_occ")
rej = ", ".join(f"Länge {i + 1}: {ex.rejected_by_span[i]} von {ex.requested_by_span[i]}" for i in range(L))
st.caption(f"Ertrag in dieser Folge: {_num(ex.revenue)}. Abgelehnt (aus Platzmangel oder von der Regel), je Länge der Fahrt: {rej}.")

# ---------------------------------------------------------------------------------------------- Kernfrage 1
st.markdown("## Kernfrage 1: Was kostet die Annahme nach Eingang?")
_show(verdict_fcfs(a))
rows1 = []
for key in ("fcfs", "dp"):
    s = a.rules[key]
    rows1.append({"Regel": C.RULE_LABELS[key] if key == "fcfs" else "Exakte Rechnung", f"angenommen: Länge 1": _pct(s.acc_short), f"angenommen: {w['long']}": _pct(s.acc_long)})
st.dataframe(rows1, hide_index=True, width="stretch")
st.caption(f"Anteil der angefragten Fahrten, der angenommen wird (alle {a.n} Folgen); {w['long']} sind die mit mehr als der halben Linie ({a.n_long_products} von {a.n_products} Produkten). "
           f"Je Folge werden im Mittel {_num(a.requested_short)} Anfragen der Länge 1 und {_num(a.requested_long)} Anfragen nach {w['long_dat']} gestellt.")

# ---------------------------------------------------------------------------------------------- Kernfrage 2
st.markdown("## Kernfrage 2: Welche Regel reicht?")
_show(verdict_rules(a))
rows2 = []
for key in ("fcfs", "bid_static", "bid_resolve") + (("dp_hat",) if params.err != 0.0 else ()):
    s = a.rules[key]
    rows2.append({"Regel": C.RULE_LABELS.get(key, "Exakte Rechnung mit der falschen Prognose"), "Ertrag in % der exakten Rechnung": _num(100 + s.gap.mean), "Standardfehler (Prozentpunkte)": _num(s.gap.se),
                  "exakter Erwartungswert in %": "—" if s.exact_gap is None else _num(100 + s.exact_gap),
                  "Unterschied zur Annahme nach Eingang (Prozentpunkte)": "—" if key == "fcfs" else f"{_sgn(s.vs_fcfs.mean)} ± {_num(s.vs_fcfs.se)}"})
st.dataframe(rows2, hide_index=True, width="stretch")
st.caption("Alle Regeln auf denselben Anfragefolgen; „±“ ist der Standardfehler der gepaarten Differenz. Die Neuberechnung hat keinen exakten Erwartungswert (das LP hängt am Zustand und an der Zeit): nur Simulation.")

# ---------------------------------------------------------------------------------------------- Kernfrage 3
st.markdown("## Kernfrage 3: Wie genau muss die Prognose sein?")
_show(verdict_forecast(a))
fs = forecast_summary(DATA["prognose"])
rows3 = []
for r in DATA["prognose"]:
    rows3.append({"Szenario": f"{r['szenario']}: {r['label']}", "Prognose für lange Fahrten": f"{100 * r['err']:+.0f} %".replace("-", "−"), "Annahme nach Eingang": _num(100 + r["fcfs"]["gap"]),
                  "Bid-Price einmal": _num(100 + r["bid_static"]["gap"]), "Bid-Price neu": _num(100 + r["bid_resolve"]["gap"]), "exakte Rechnung mit Prognose": _num(100 + r["dp_hat"]["gap"]),
                  "exakter Erwartungswert": _num(100 + r["dp_hat"]["exact_gap"])})
st.dataframe(rows3, hide_index=True, width="stretch", height=300)
st.caption(f"Vorgerechnet (Netz mit 4 Teilstrecken und 8 Plätzen je Teilstrecke, {DATA['prognose'][0]['n']} Folgen je Zeile): Ertrag in % der exakten Rechnung mit wahrer Nachfrage. "
           f"Die exakte Rechnung mit verzerrter Prognose verliert nach dem exakten Erwartungswert höchstens {_num(fs['dp_hat_worst'], 2)} % (größter Verlust über alle {fs['rows']} Zeilen).")

# ---------------------------------------------------------------------------------------------- Experimente
st.markdown("## Experimente (vorgerechnet)")
st.caption("Anfragefolgen mit festen Zufalls-Seeds, jede Zelle mit eigenem Zufalls-Seed und exakter Rechnung als Maßstab (`tools/make_results.py`); nichts davon rechnet beim Öffnen der Seite.")
RAST = DATA["raster"]
rs = raster_summary(RAST)
a1r = alpha_one_resolve(DATA["prognose"])

st.subheader("🔬 1 · Wie viel kostet die Annahme nach Eingang, je nach Last und Preisgefälle?")
st.dataframe([{"Plätze je Teilstrecke": r["cap"], "Last": _num(r["rho"]), "Preisgefälle α": _num(r["alpha"]), "Buchungszeit": EARLY_LABELS[r["early"]],
               "Annahme nach Eingang (% der exakten Rechnung)": _num(100 + r["fcfs"]["gap"]), "Bid-Price einmal": _num(100 + r["bid_static"]["gap"]),
               "Bid-Price minus Eingang (Prozentpunkte)": f"{_sgn(r['bid_vs_fcfs'])} ± {_num(r['bid_vs_fcfs_se'])}"} for r in RAST], hide_index=True, width="stretch", height=360)
st.caption(f"{rs['cells']} Zellen (Zug mit 4 Teilstrecken), je {RAST[0]['n']} Folgen. Annahme nach Eingang erreicht {_num(100 + rs['fcfs_min'])} bis {_num(100 + rs['fcfs_max'])} % der exakten Rechnung; bei Last 0,8 mindestens {_num(100 + rs['low_load_fcfs_min'])} %; "
           f"in den {rs['over_cells']} Zellen mit Last ab 2 und α höchstens 0,6 zwischen {_num(100 + rs['over_fcfs_min'])} und {_num(100 + rs['over_fcfs_max'])} %. Der einmal gerechnete Bid-Price erreicht dort {_num(100 + rs['over_bid_min'])} bis {_num(100 + rs['over_bid_max'])} %.")

st.subheader("🔬 2 · Gleicher Preis je Teilstrecke: der einmal gerechnete Bid-Price ändert nichts")
st.dataframe([{"Plätze je Teilstrecke": r["cap"], "Last": _num(r["rho"]), "Buchungszeit": EARLY_LABELS[r["early"]], "Annahme nach Eingang": _num(100 + r["fcfs"]["gap"]),
               "Bid-Price einmal": _num(100 + r["bid_static"]["gap"]), "dieselben Anfragen angenommen": "ja" if r["bid_same_as_fcfs"] else "nein"} for r in RAST if r["alpha"] == 1.0],
             hide_index=True, width="stretch", height=300)
st.caption(f"Die {rs['a1_cells']} Zellen mit α = 1 aus Experiment 1: In {'allen' if rs['a1_same'] == rs['a1_cells'] else rs['a1_same']} davon nimmt der einmal gerechnete Bid-Price in allen Folgen genau dieselben Anfragen an wie die Annahme nach Eingang. "
           f"Die Lücke zur exakten Rechnung bleibt ({_num(100 + rs['a1_fcfs_min'])} bis {_num(100 + rs['a1_fcfs_max'])} %). Gemessen ist hier nur der einmal gerechnete Bid-Price; die Neuberechnung je Periode erreicht im Prognose-Szenario D (α = 1, Last 1,2) "
           f"{_num(100 + a1r['bid_resolve_min'])} bis {_num(100 + a1r['bid_resolve_max'])} % gegen {_num(100 + a1r['fcfs_min'])} bis {_num(100 + a1r['fcfs_max'])} % für die Annahme nach Eingang.")

st.subheader("🔬 3 · Wann ist der einmal gerechnete Bid-Price schlechter als die Annahme nach Eingang?")
worse = rs["worse"]
st.dataframe([{"Plätze je Teilstrecke": r["cap"], "Last": _num(r["rho"]), "Preisgefälle α": _num(r["alpha"]), "Buchungszeit": EARLY_LABELS[r["early"]], "Bid-Price minus Eingang (Prozentpunkte)": f"{_sgn(r['bid_vs_fcfs'])} ± {_num(r['bid_vs_fcfs_se'])}"} for r in worse],
             hide_index=True, width="stretch")
st.caption(f"Zellen aus Experiment 1, in denen der Unterschied mehr als zwei Standardfehler unter null liegt: {len(worse)} von {rs['cells']}, alle bei Last {', '.join(sorted({_num(r['rho']) for r in worse}))} und kleinen Kapazitäten. "
           f"In {rs['better']} Zellen ist der Bid-Price um mehr als zwei Standardfehler besser, in {rs['same_cells']} Zellen entscheidet er in allen Folgen wie die Annahme nach Eingang. "
           f"Nach dem exakten Erwartungswert beider Regeln (ohne Simulation) ist er in {rs['worse_exact']} Zellen schlechter.")

st.subheader("🔬 4 · Wächst die Lücke mit der Größe des Netzes?")
NET = DATA["netz"]
st.dataframe([{"Teilstrecken": r["L"], "Produkte": r["produkte"], "Zustände": _thousands(r["zustaende"]), "Annahme nach Eingang": _num(100 + r["fcfs"]["gap"]), "Bid-Price einmal": _num(100 + r["bid_static"]["gap"]),
               "Bid-Price neu": _num(100 + r["bid_resolve"]["gap"])} for r in NET], hide_index=True, width="stretch")
st.caption(f"Ertrag in % der exakten Rechnung, {NET[0]['n']} Folgen je Zeile, Last 2, α 0,3, 6 Plätze je Teilstrecke, lange Fahrten früh. Die Zahl der Zustände wächst mit (Plätze + 1) hoch Teilstrecken. "
           "Zustände und Speicher der exakten Rechnung:")
st.dataframe([{"Teilstrecken": r["L"], "Plätze je Teilstrecke": r["cap"], "Zustände": _thousands(r["zustaende"]), "Perioden": r["T"], "Speicher für alle Perioden (MB)": _num(r["mb"], 0), "live in dieser App": "ja" if r["live"] else "nein"}
              for r in DATA["zustaende"]], hide_index=True, width="stretch")

st.subheader("🔬 5 · Ein fester Schutz (Protection Level) für lange Fahrten")
SCH = DATA["schutz"]
st.dataframe([{"Szenario": f"{r['szenario']}: {r['label']}", "bester Schutz k": r["best_k"], "Annahme nach Eingang": _num(100 + r["fcfs"]["gap"]), "mit festem Schutz": _num(100 + r["schutz"]["gap"]),
               "Bid-Price einmal": _num(100 + r["bid_static"]["gap"])} for r in SCH], hide_index=True, width="stretch")
st.caption(f"Regel: eine lange Fahrt wird nur angenommen, wenn danach auf jeder ihrer Teilstrecken noch k Plätze frei bleiben (k = 8 heißt: keine lange Fahrt annehmen). k wird aus 0 bis 8 auf {SCH[0]['n_train']} Trainingsfolgen gewählt und auf {SCH[0]['n_test']} frischen Folgen gemessen "
           "(Netz mit 4 Teilstrecken und 8 Plätzen je Teilstrecke). Ertrag in % der exakten Rechnung.")

st.subheader("🔬 6 · Dasselbe Modell im Hotel")
HOT = DATA["hotel"]
st.dataframe([{"Last": _num(r["rho"]), "Preisgefälle α": _num(r["alpha"]), "Annahme nach Eingang": _num(100 + r["fcfs"]["gap"]), "Bid-Price einmal": _num(100 + r["bid_static"]["gap"]),
               "Bid-Price minus Eingang (Prozentpunkte)": f"{_sgn(r['bid_vs_fcfs'])} ± {_num(r['bid_vs_fcfs_se'])}"} for r in HOT], hide_index=True, width="stretch")
st.caption(f"Hotel mit 5 Nächten und 8 Zimmern je Nacht, Aufenthalte früh gebucht, {HOT[0]['n']} Folgen je Zeile; Ertrag in % der exakten Rechnung. Ein Aufenthalt von s Nächten zahlt 10 · s hoch α.")

# ---------------------------------------------------------------------------------------------- Expander
with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    tab_methods, tab_exact = st.tabs(["📊 Regeln", "🧮 Exakt (Rechnung neu starten)"])
    with tab_methods:
        st.markdown(
            """
| Regel | Eine Anfrage wird angenommen, wenn sie passt und … |
|---|---|
| **Annahme nach Eingang** | … immer |
| **Bid-Price, einmal gerechnet** | … ihr Preis mindestens die Summe der Schattenpreise ihrer Teilstrecken deckt; die Schattenpreise sind die Duale des Optimierungsproblems „größter Ertrag bei erwarteter Nachfrage und Kapazität“, einmal zu Beginn gelöst |
| **Bid-Price, je Periode neu** | … dasselbe, aber das Problem wird in jeder Periode mit den freien Plätzen und der Restnachfrage neu gelöst |
| **Exakte Rechnung** | … der Preis plus der Wert der danach verbleibenden Belegung mindestens so groß ist wie der Wert der jetzigen Belegung (Rückwärtsrekursion über alle Belegungen) |

Alle Regeln sehen dieselben Anfragefolgen; Unterschiede sind gepaarte Differenzen. Alle Zufallsfolgen kommen aus einem eigenen SplitMix64. """ + f"Die Live-Rechnung nutzt {C.LIVE_SEQ} Folgen, die Messreihen {DATA['raster'][0]['n']} (Neuberechnung: {DATA['netz'][0]['n']})." + """
"""
        )
    with tab_exact:
        st.markdown("Rechnet die **exakte Rechnung** für das eingestellte Netz neu und zeigt Erwartungswert, Zahl der Zustände, Speicher und Rechenzeit.")
        if st.button("Exakte Rechnung starten", key="exact_run"):
            net0 = build_net(params)
            t1 = time.perf_counter()
            V0 = A.dp_values(net0, net0.lam)
            secs = time.perf_counter() - t1
            st.session_state["exact_result"] = (params, float(V0[0][(net0.cap,) * net0.L]), net0.T, secs)
            del V0
        stored = st.session_state.get("exact_result")
        if stored and stored[0] == params:
            _p, value, T, secs = stored
            st.metric("Erwarteter Ertrag der exakten Rechnung", _num(value, 2))
            st.caption(f"{_thousands(n_states(params.L, params.cap))} Zustände, {T} Perioden, Speicher für alle Perioden {_num((T + 1) * n_states(params.L, params.cap) * 8 / 1e6, 0)} MB, Rechenzeit {secs:.2f} s.")
        elif stored:
            st.caption("Die Einstellungen haben sich geändert: erneut „Exakte Rechnung starten“.")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/teilstrecken-buchung-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind eine Linie mit $L$ Teilstrecken, je $K$ Plätze, Produkte (Fahrten) mit Preisen $f_p$ und Anfragen mit Wahrscheinlichkeiten $\lambda_{pt}$. Gesucht sind Annahmeregeln und ihr erwarteter Ertrag im Vergleich zur besten Regel. Die erste Formel berechnet die beste Regel rückwärts über alle Belegungen, die zweite das Optimierungsproblem, dessen Duale die Schattenpreise liefern.

| Symbol | Bedeutung |
|---|---|
| $L, K$ | Zahl der Teilstrecken, Plätze je Teilstrecke |
| $a_{\ell p}, f_p$ | 1, wenn Produkt $p$ die Teilstrecke $\ell$ belegt, sonst 0; Preis des Produkts |
| $\lambda_{pt}$ | Wahrscheinlichkeit, dass in Periode $t$ Produkt $p$ angefragt wird |
| $V_t(x)$ | größter erwarteter Ertrag ab Periode $t$ bei freien Plätzen $x$ (ein Wert je Teilstrecke) |
| $\pi_\ell, D_p$ | Schattenpreis der Teilstrecke $\ell$, erwartete Gesamtnachfrage nach Produkt $p$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/teilstrecken-buchung-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Gegeben:** Eine Linie mit $L$ Teilstrecken $\ell = 1, \dots, L$, auf jeder $K$ Plätze. Ein **Produkt** $p = (i, j)$ mit $0 \le i < j \le L$ belegt je einen Platz auf den Teilstrecken $i+1, \dots, j$ ($a_{\ell p} = 1$, sonst $0$),
hat die Länge $s_p = j - i$ und den Preis $f_p = 10 \cdot s_p^{\alpha}$. In jeder Periode $t = 1, \dots, T$ kommt höchstens eine Anfrage, nach Produkt $p$ mit der Wahrscheinlichkeit $\lambda_{pt}$ (mit $\sum_p \lambda_{pt} \le 0{,}98$).
Die Nachfrage je Produkt der Länge $s$ ist proportional zu $e^{-\gamma (s-1)}$ ($\gamma = 0{,}5$ im Zug, $0{,}8$ im Hotel) und so skaliert, dass die erwartete Nachfrage je Teilstrecke im Mittel $\rho K$ beträgt ($\rho$: Last); $T = \lceil \sum_p D_p / 0{,}5 \rceil$ mit $D_p = \sum_t \lambda_{pt}$.
Der zeitliche Verlauf eines Produkts ist $1 + e\,\sigma_p\,(1 - 2\tau)$ mit $\tau = (t - \tfrac12)/T$, $\sigma_p = (s_p - \bar s)/(s_{\max} - \bar s)$ ($\bar s$: Mittel der Längen aller Produkte) und der Buchungszeit $e \in \{0; 0{,}5; 1\}$, nach unten bei $0{,}05$ gekappt und je Produkt auf Mittel 1 normiert.

**Gesucht:** eine Regel, die in Periode $t$ im Zustand $x_t \in \{0, \dots, K\}^L$ (freie Plätze je Teilstrecke) die Anfrage $p_t$ annimmt ($u_t = 1$) oder ablehnt ($u_t = 0$) und den erwarteten Ertrag $\mathbb{E}\bigl[\sum_t f_{p_t} u_t\bigr]$ maximiert; $u_t = 1$ ist nur zulässig, wenn $x_t \ge a_{p_t}$, und der Zustand folgt $x_{t+1} = x_t - u_t\, a_{p_t}$, $x_1 = (K, \dots, K)$.

**Exakte Rechnung (Dynamische Programmierung):** $V_{T+1} \equiv 0$ und für $t = T, \dots, 1$ und jeden Zustand $x$

$$V_t(x) = V_{t+1}(x) + \sum_p \lambda_{pt}\, \max\bigl\{0,\; f_p + V_{t+1}(x - a_p) - V_{t+1}(x)\bigr\}\;\;\text{(Summand nur für } x \ge a_p\text{)}$$

Die Regel nimmt $p$ in $x$ an, wenn $x \ge a_p$ und $f_p + V_{t+1}(x - a_p) \ge V_{t+1}(x)$ gilt. $V_1(K, \dots, K)$ ist der größte erwartete Ertrag. Es gibt $(K+1)^L$ Zustände.

**Bid-Price:** Das Optimierungsproblem

$$\max \sum_p f_p\, y_p \quad\text{unter}\quad \sum_p a_{\ell p}\, y_p \le K \;(\ell = 1, \dots, L),\;\; 0 \le y_p \le D_p$$

hat zu den Kapazitätsbedingungen die Dualvariablen $\pi_\ell \ge 0$ (Schattenpreise). Die Regel nimmt $p$ an, wenn es passt und $f_p \ge \sum_\ell a_{\ell p}\, \pi_\ell$ gilt. Bei der Neuberechnung in Periode $t$ steht $x_{t\ell}$ statt $K$ und $\sum_{\tau \ge t} \lambda_{p\tau}$ statt $D_p$.

**Prognose:** Die Regeln rechnen mit $\hat\lambda_{pt} = (1 + \varepsilon)\,\lambda_{pt}$ für lange Produkte ($2 s_p > L$) und $\hat\lambda_{pt} = \lambda_{pt}$ sonst; erzeugt wird die Folge mit $\lambda$. Die Referenz rechnet mit $\lambda$.

**Kennzahl:** Ertrag in % der exakten Rechnung $= 100 \cdot \bar R_{\text{Regel}} / \bar R_{\text{exakt}}$, die Mittel über dieselben Folgen.
"""
    )

with st.expander("🚧 Wo die Annahmen enden"):
    st.markdown(
        """
- **Nachfrage, Preise und Buchungszeit** sind Zufallsmodelle mit gesetzten Parametern (Last, α, Buchungszeit), nicht an Buchungsdaten kalibriert. Die Prozentzahlen gelten für diese Modelle, nicht für echte Bahnen oder Hotels.
- Es gibt **einen Preis je Produkt**, keine Preisklassen, keine Stornierungen und No-Shows, keine Überbuchung, kein Umbuchen und keine Sitzplatzzuweisung: Geprüft wird die Kapazität je Teilstrecke.
- Die **exakte Rechnung** ist ein Maßstab für den bestmöglichen Ertrag einer sofort entscheidenden Regel; sie rechnet nur bei kleinen Netzen (hier höchstens """ + f"{_thousands(C.STATES_LIVE_MAX)}" + """ Zustände live).
- **Bid-Price** ist ein gängiges Verfahren des Revenue Managements in Netzen (Talluri und van Ryzin 1998); hier ist es in der einfachsten Form umgesetzt. Die Aussagen gelten für diese Umsetzung und die gemessenen Zellen, nicht für jede Bid-Price-Variante.
- Die **Prognosefehler** verzerren nur die Nachfrage nach langen Fahrten, um höchstens 50 %, in vier Szenarien; andere Fehler (Preise, kurze Fahrten, Zeitverlauf) sind nicht gemessen.
- Ein Zug oder ein Hotel mit nur einer Linie; Netze mit Verzweigungen sind nicht umgesetzt.
"""
    )
    st.markdown("**Annahmen im Überblick**")
    st.markdown(ASSUMPTIONS_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zum Thema: [Sitzplätze optimieren](https://sebastianhanisch.net/sitzplaetze-optimieren.html)."
)
