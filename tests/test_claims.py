"""Jede Zahl der Befunde in der README wird hier nachgerechnet (DEMO-PLAYBOOK §4, §11a): aus den vorgerechneten Messreihen (`data/tsb_results.json`) über dieselben Auswertungsfunktionen wie in der App,
aus der Live-Analyse der Voreinstellung, und es wird geprüft, dass die Zahl in der README steht. Die Messreihen selbst sind Messungen (feste Zufalls-Seeds); der Test `test_one_cell_is_reproducible`
rechnet eine Zelle neu und vergleicht sie mit der gespeicherten."""

from pathlib import Path

import pytest

import tsb_evaluation as ev
from tsb_scenario import Params

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
DATA = ev.load_results()


def f1(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def pct(x: float) -> str:
    return f1(100.0 + x)


def in_readme(text: str):
    assert text in README, text


def test_measurement_sizes_match_the_readme_description():
    assert len(DATA["raster"]) == 72 and {r["n"] for r in DATA["raster"]} == {1000}
    assert len(DATA["hotel"]) == 9 and {r["n"] for r in DATA["hotel"]} == {1000}
    assert len(DATA["prognose"]) == 20 and {r["n"] for r in DATA["prognose"]} == {500}
    assert len(DATA["netz"]) == 5 and {r["n"] for r in DATA["netz"]} == {500}
    assert [(r["n_train"], r["n_test"]) for r in DATA["schutz"]] == [(500, 1000)] * 4
    assert sorted({(r["cap"], r["rho"], r["alpha"], r["early"]) for r in DATA["raster"]}) == sorted(
        (c, p, a, e) for c in (4, 8, 16) for p in (0.8, 1.2, 2.0, 3.0) for a in (1.0, 0.6, 0.3) for e in (0.0, 1.0))
    assert len({r["T"] for r in DATA["raster"] if r["cap"] == 4}) > 1                       # die Zellen sind verschiedene Szenarien, keine Wiederholung


def test_raster_fcfs():
    s = ev.raster_summary(DATA["raster"])
    in_readme(f"**{pct(s['fcfs_min'])} bis {pct(s['fcfs_max'])} %**")
    in_readme(f"mindestens **{pct(s['low_load_fcfs_min'])} %**")
    assert s["over_cells"] == 24
    in_readme(f"**24 Zellen** mit Last ab 2 und α höchstens 0,6 zwischen **{pct(s['over_fcfs_min'])} und {pct(s['over_fcfs_max'])} %**")
    assert max(r["fcfs"]["se"] for r in DATA["raster"]) <= 0.34
    in_readme("höchstens 0,34 Prozentpunkte")


def test_raster_bid_price():
    s = ev.raster_summary(DATA["raster"])
    in_readme(f"**{pct(s['bid_min'])} bis {pct(s['bid_max'])} %**")
    in_readme(f"zwischen **{pct(s['over_bid_min'])} und {pct(s['over_bid_max'])} %**")
    assert max(r["bid_static"]["se"] for r in DATA["raster"]) <= 0.27


def test_alpha_one_is_identical():
    s = ev.raster_summary(DATA["raster"])
    assert s["a1_cells"] == 24 and s["a1_same"] == 24
    assert all(r["bid_vs_fcfs"] == 0.0 and r["bid_vs_fcfs_se"] == 0.0 for r in DATA["raster"] if r["alpha"] == 1.0)      # auch die Erträge sind gleich
    in_readme(f"(**{pct(s['a1_fcfs_min'])} bis {pct(s['a1_fcfs_max'])} %**)")
    others = [r for r in DATA["raster"] if r["alpha"] != 1.0 and r["bid_same_as_fcfs"]]
    assert len(others) == 12 and {r["rho"] for r in others} == {0.8} and {r["alpha"] for r in others} == {0.6, 0.3}
    assert all(r["bid_same_as_fcfs"] for r in DATA["raster"] if r["rho"] == 0.8)                  # bei Last 0,8 in allen 24 Zellen
    assert s["same_cells"] == 36
    in_readme("zusammen in **36 von 72 Zellen** gleiche Entscheidungen")
    r1 = ev.alpha_one_resolve(DATA["prognose"])
    assert r1["rows"] == 5 and r1["resolve_beats_fcfs"]
    in_readme(f"erreicht sie **{pct(r1['bid_resolve_min'])} bis {pct(r1['bid_resolve_max'])} %** gegen **{pct(r1['fcfs_min'])} bis {pct(r1['fcfs_max'])} %** für die Annahme nach Eingang")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import make_results as mk

    assert mk.SCENARIOS["D"][1] == dict(rho=1.2, alpha=1.0, early=0.0)                  # Szenario D der Prognose-Messreihe: Netz mit 4 Teilstrecken und 8 Plätzen, Last 1,2, α 1


def test_bid_price_worse_than_first_come():
    s = ev.raster_summary(DATA["raster"])
    worse = s["worse"]
    assert len(worse) == 7 and {r["rho"] for r in worse} == {1.2} and {r["alpha"] for r in worse} <= {0.6, 0.3}
    by_cap = {cap: [r["bid_vs_fcfs"] for r in worse if r["cap"] == cap] for cap in (4, 8, 16)}
    assert len(by_cap[4]) == 4 and len(by_cap[8]) == 3 and by_cap[16] == []
    in_readme("In **7 von 72 Zellen**")
    # nach dem exakten Erwartungswert beider Regeln: 8 Zellen, die achte mit −0,2 Prozentpunkten (8 Plätze, Last 1,2, α 0,3, früh); alle bei Last 1,2, keine mit 16 Plätzen
    ex = [r for r in DATA["raster"] if r["bid_static"]["exact_gap"] < r["fcfs"]["exact_gap"] - 1e-9]
    assert s["worse_exact"] == len(ex) == 8 and {r["rho"] for r in ex} == {1.2} and 16 not in {r["cap"] for r in ex}
    eighth = max(ex, key=lambda r: r["bid_static"]["exact_gap"] - r["fcfs"]["exact_gap"])
    assert (eighth["cap"], eighth["alpha"], eighth["early"]) == (8, 0.3, 1.0) and f1(eighth["bid_static"]["exact_gap"] - eighth["fcfs"]["exact_gap"]) == "-0,2"
    in_readme("in **8 Zellen** schlechter, die achte (8 Plätze, Last 1,2, α 0,3, früh) mit −0,2 Prozentpunkten")
    in_readme(f"bei 4 Plätzen je Teilstrecke {f1(max(by_cap[4])).replace('-', '−')} bis {f1(min(by_cap[4])).replace('-', '−')}")
    in_readme(f"bei 8 Plätzen {f1(max(by_cap[8])).replace('-', '−')} bis {f1(min(by_cap[8])).replace('-', '−')} Prozentpunkte")
    assert s["better"] == 28
    in_readme(f"In **28 Zellen** ist er um mehr als zwei Standardfehler besser (bis +{f1(s['bid_vs_fcfs_max'])} Prozentpunkte)")


def test_default_numbers():
    a = ev.analyse(ev.Settings())
    fc, bid, res = a.rules["fcfs"], a.rules["bid_static"], a.rules["bid_resolve"]
    in_readme(f"Annahme nach Eingang **{pct(fc.exact_gap)} %**, Bid-Price einmal gerechnet **{pct(bid.exact_gap)} %** (der exakte Ertrag der exakten Rechnung ist {f1(a.v0)})")
    in_readme(f"Simulation mit 200 Folgen: {pct(fc.gap.mean)} / {pct(bid.gap.mean)} / {pct(res.gap.mean)} %")
    in_readme(f"{100 * fc.acc_short:.0f} % an, von den langen {100 * fc.acc_long:.0f} %; die exakte Rechnung {100 * a.rules['dp'].acc_short:.0f} % und {100 * a.rules['dp'].acc_long:.0f} %")
    in_readme(f"Standardfehler {f1(fc.gap.se)} / {f1(bid.gap.se)} / {f1(res.gap.se)} Prozentpunkte")
    assert a.n == 200 and a.settings.seed == 35 and a.settings.params == Params()


def test_neuberechnung():
    prog = DATA["prognose"]
    in_readme(f"**{pct(min(r['bid_resolve']['gap'] for r in prog))} bis {pct(max(r['bid_resolve']['gap'] for r in prog))} %** (einmal gerechnet: {pct(min(r['bid_static']['gap'] for r in prog))} bis {pct(max(r['bid_static']['gap'] for r in prog))} %)")
    net = DATA["netz"]
    in_readme(f"in der Netzgrößen-Messreihe {pct(min(r['bid_resolve']['gap'] for r in net))} bis {pct(max(r['bid_resolve']['gap'] for r in net))} %")
    assert len({r["szenario"] for r in prog}) == 4 and sorted({r["err"] for r in prog}) == [-0.5, -0.25, 0.0, 0.25, 0.5]


def test_forecast():
    prog = DATA["prognose"]
    fs = ev.forecast_summary(prog)
    in_readme(f"**höchstens {fs['dp_hat_worst']:.2f} %**".replace(".", ","))
    for key in "AC":
        assert max(-r["dp_hat"]["exact_gap"] for r in prog if r["szenario"] == key) < 0.05
    in_readme("in den Szenarien A und C (Überlast, lange Fahrten billig) weniger als 0,05 %")
    assert all(r["dp_hat"]["exact_gap"] <= 1e-9 for r in prog)                         # die falsche Prognose gewinnt nie gegen die richtige (Erwartungswert)
    assert all(abs(r["dp_hat"]["exact_gap"]) < 1e-9 for r in prog if r["err"] == 0.0)


def test_net_size():
    net = DATA["netz"]
    assert [r["L"] for r in net] == [2, 3, 4, 5, 6]
    in_readme(f"**{pct(min(r['fcfs']['gap'] for r in net))} bis {pct(max(r['fcfs']['gap'] for r in net))} %**")
    bids = [pct(r["bid_static"]["gap"]) for r in net]
    in_readme(f"steigt nicht gleichmäßig ({' / '.join(bids)} %)")
    from_three = [100 + r["bid_static"]["gap"] for r in net if r["L"] >= 3]
    assert max(from_three) < 100 + net[0]["bid_static"]["gap"]                         # ab 3 Teilstrecken unter dem Wert bei 2
    in_readme(f"ab 3 Teilstrecken mit {f1(min(from_three))} bis {f1(max(from_three))} % unter dem Wert bei 2 Teilstrecken ({bids[0]} %)")
    gaps = [r["bid_static"]["gap"] for r in net]
    assert not all(a <= b for a, b in zip(gaps, gaps[1:])) and not all(a >= b for a, b in zip(gaps, gaps[1:]))          # tatsächlich nicht monoton
    assert [r["zustaende"] for r in net] == [49, 343, 2401, 16807, 117649]


def test_protection():
    sch = {r["szenario"]: r for r in DATA["schutz"]}
    for key in "AC":
        r = sch[key]
        assert r["best_k"] == 8                                 # k = Kapazität: keine lange Fahrt annehmen; die Suche lief über 0 bis 8, nicht nur bis 7
        assert r["schutz"]["gap"] > r["fcfs"]["gap"] and r["bid_static"]["gap"] > r["schutz"]["gap"]
    assert sch["B"]["best_k"] == 0 and sch["D"]["best_k"] == 0
    assert sch["B"]["schutz"]["gap"] == sch["B"]["fcfs"]["gap"] and sch["D"]["schutz"]["gap"] == sch["D"]["fcfs"]["gap"]
    in_readme(f"in Szenario A {pct(sch['A']['schutz']['gap'])} % (k = 8; Annahme nach Eingang {pct(sch['A']['fcfs']['gap'])} %), in C {pct(sch['C']['schutz']['gap'])} % (k = 8; {pct(sch['C']['fcfs']['gap'])} %), "
              f"der einmal gerechnete Bid-Price {pct(sch['A']['bid_static']['gap'])} und {pct(sch['C']['bid_static']['gap'])} %")
    in_readme("k = 0 bis 8 gewählten k (k = 8 heißt: keine lange Fahrt annehmen)")
    in_readme("In B und D ist k = 0 am besten")


def test_hotel():
    hot = DATA["hotel"]
    a1 = [r for r in hot if r["alpha"] == 1.0]
    assert len(a1) == 3 and all(r["bid_same_as_fcfs"] for r in a1)
    r = next(r for r in hot if r["rho"] == 3.0 and r["alpha"] == 0.3)
    in_readme(f"erreicht die Annahme nach Eingang **{pct(r['fcfs']['gap'])} %**, der Bid-Price **{pct(r['bid_static']['gap'])} %**")
    assert all(x["T"] > 0 for x in hot)


def test_simulation_check():
    ras = DATA["raster"]
    n_off = sum(1 for r in ras if abs(r["sim_vs_v0"]) > 2 * r["sim_vs_v0_se"])
    assert n_off == 3
    in_readme("In **3 von 72** Rasterzellen")
    off = [r for r in ras if abs(r["sim_vs_v0"]) > 2 * r["sim_vs_v0_se"]]
    assert max(abs(r["sim_vs_v0"]) for r in off) < 0.07                                 # in diesen drei Zellen ist der Standardfehler fast null (16 Plätze, α = 1)
    assert all(r["cap"] == 16 and r["alpha"] == 1.0 and r["sim_vs_v0_se"] < 0.04 for r in off)
    in_readme("dort sind es höchstens 0,07 %, weil der Standardfehler fast null ist")
    biggest = max(ras, key=lambda r: abs(r["sim_vs_v0"]))
    assert abs(biggest["sim_vs_v0"]) <= 2 * biggest["sim_vs_v0_se"]
    in_readme(f"Die größte Abweichung über alle Zellen ist **{abs(biggest['sim_vs_v0']):.2f} %** und liegt innerhalb von zwei Standardfehlern".replace(".", ","))


def test_vormessung_statement_matches_the_current_measurement():
    s = ev.raster_summary(DATA["raster"])
    assert f1(-s["fcfs_min"]) == "25,8" and f1(-s["bid_min"]) == "4,4" and len(s["worse"]) == 7
    in_readme("sind es −25,8 %, −4,4 % und sieben Zellen")


def test_one_cell_is_reproducible():
    """Dieselbe Zelle (Raster, Zelle 0: 4 Plätze, Last 0,8, α 1, gleichmäßig) mit demselben Zufalls-Seed ergibt dieselben Werte wie die gespeicherte."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import make_results as mk

    stored = DATA["raster"][0]
    row = mk.fast_cell(Params(L=4, cap=stored["cap"], rho=stored["rho"], alpha=stored["alpha"], early=stored["early"]), mk.BASE["raster"], mk.N_FAST)
    assert row["fcfs"]["gap"] == pytest.approx(stored["fcfs"]["gap"], abs=1e-9) and row["bid_static"]["gap"] == pytest.approx(stored["bid_static"]["gap"], abs=1e-9)
    assert row["v0"] == pytest.approx(stored["v0"], abs=1e-9)


def test_state_table_matches_the_arithmetic():
    for r in DATA["zustaende"]:
        assert r["zustaende"] == (r["cap"] + 1) ** r["L"] and r["live"] == (r["zustaende"] <= 120_000)
    s6 = next(r for r in DATA["zustaende"] if (r["L"], r["cap"]) == (6, 8))
    assert s6["zustaende"] == 531441 and not s6["live"]
    in_readme("bei 117.649 Zuständen rund 0,3 Sekunden, bei 531.441 rund 7 Sekunden und 455 MB")
    assert round(s6["mb"]) == 455
