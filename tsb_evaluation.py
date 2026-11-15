"""Auswertung: Live-Analyse (Anfragefolgen, Regeln, Abstand zur exakten Rechnung), Verdikte als reine Funktionen, Zugriff auf die vorgerechneten Messreihen."""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import tsb_algorithm as A
import tsb_constants as C
from tsb_rng import SplitMix64
from tsb_scenario import Params, build_net, n_states, sample_sequence

RESULTS_PATH = Path(__file__).resolve().parent / "data" / "tsb_results.json"
ANALYSIS_RULES = ("fcfs", "bid_static", "bid_resolve", "dp_hat", "dp")      # dp_hat = exakte Rechnung mit der (verzerrten) Prognose, dp = mit der wahren Nachfrage (Referenz)


@dataclass(frozen=True, kw_only=True)
class Settings:
    params: Params = Params()
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Stat:
    mean: float
    se: float


@dataclass(frozen=True)
class RuleStat:
    revenue: float            # mittlerer Ertrag über die Folgen
    gap: Stat                 # Abstand zur exakten Rechnung in % ihres Ertrags (negativ = weniger), Standardfehler aus gepaarten Differenzen
    vs_fcfs: Stat             # Unterschied zur Annahme nach Eingang in Prozentpunkten des Ertrags der exakten Rechnung
    exact_gap: float | None   # dasselbe als exakter Erwartungswert ohne Simulation (nicht für die Neuberechnung)
    acc_short: float | None   # Anteil angenommener Fahrten der Länge 1 an den angefragten (alle Folgen), None ohne Anfragen
    acc_long: float | None    # dasselbe für lange Fahrten (mehr als die halbe Linie)
    same_as_fcfs: bool        # in allen Folgen dieselben Anfragen angenommen wie bei der Annahme nach Eingang


@dataclass(frozen=True)
class Example:
    rule: str
    events: tuple             # (Periode, Produkt, angenommen?)
    occupancy: tuple          # Belegung je Teilstrecke und Fahrtlänge (Tupel von Tupeln)
    revenue: float
    rejected_by_span: tuple   # abgelehnte Anfragen je Fahrtlänge (Länge 1 bis L)
    requested_by_span: tuple


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    n: int
    T: int
    n_products: int
    states: int
    v0: float                 # exakter Erwartungswert der exakten Rechnung bei wahrer Nachfrage
    sim_vs_v0: Stat           # simulierter Mittelwert der exakten Rechnung gegen v0 in % (Kontrolle der Simulation)
    rules: dict               # Regel -> RuleStat
    examples: dict            # Regel (fcfs, bid_static, bid_resolve, dp) -> Example (erste Folge)
    requested_long: float     # mittlere Zahl angefragter langer Fahrten je Folge
    requested_short: float
    n_long_products: int


def _stat_of_diff(d, base) -> Stat:
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = 100.0 * float(d.mean()) / base
    se = 100.0 * float(d.std(ddof=1)) / math.sqrt(n) / base if n > 1 else 0.0
    return Stat(mean, se)


def _class_share(acc_sum, req_sum):
    return float(acc_sum) / float(req_sum) if req_sum > 0 else None


def analyse(settings: Settings, n_seq: int = C.LIVE_SEQ) -> Analysis:
    """Live-Rechnung: n_seq (live: C.LIVE_SEQ) Anfragefolgen (Zufalls-Seed seed · 1000 + k); jede Regel auf denselben Folgen (gepaarte Vergleiche)."""
    p, seed = settings.params, settings.seed
    net = build_net(p)
    ctx_true = A.Context(net, net.lam)
    ctx_hat = ctx_true if p.err == 0.0 else A.Context(net, net.forecast(p.err))
    ctx_of = {"fcfs": ctx_true, "bid_static": ctx_hat, "bid_resolve": ctx_hat}
    v0 = float(ctx_true.V[0][(net.cap,) * net.L])
    exact = {
        "fcfs": A.value_of_policy(net, "fcfs"),
        "bid_static": A.value_of_policy(net, "bid_static", pi=ctx_hat.pi_static),
        "dp_hat": A.value_of_policy(net, "dp", V_hat=ctx_hat.V),
    }
    span_of_len = np.arange(1, net.L + 1)
    long_len = 2 * span_of_len > net.L
    rev = {r: [] for r in ANALYSIS_RULES}
    acc_tot = {r: np.zeros(net.L) for r in ANALYSIS_RULES}
    req_tot = np.zeros(net.L)
    flags = {r: [] for r in ANALYSIS_RULES}
    examples = {}
    for k in range(n_seq):
        seq = sample_sequence(net, SplitMix64(seed * C.SUBSEED_STRIDE + k))
        outs = {"dp": A.run_policy(net, seq, "dp", ctx_true)}
        for r in ("fcfs", "bid_static", "bid_resolve"):
            outs[r] = A.run_policy(net, seq, r, ctx_of[r])
        # ohne Verzerrung ist die Prognose die wahre: die exakte Rechnung mit Prognose ist dann dieselbe Rechnung
        outs["dp_hat"] = outs["dp"] if p.err == 0.0 else A.run_policy(net, seq, "dp", ctx_hat)
        for r in ANALYSIS_RULES:
            out = outs[r]
            rev[r].append(out[0])
            acc_tot[r] += out[1]
            flags[r].append(tuple(e[2] for e in out[3]))
        req_tot += outs["dp"][2]
        if k == 0:
            for r in ("fcfs", "bid_static", "bid_resolve", "dp_hat"):
                out = outs[r]
                key = "dp" if r == "dp_hat" else r                                  # Beispiel „exakte Rechnung“ = die mit der Prognose der Regeln
                occ = A.occupancy_by_span(net, out[3])
                rej = tuple(int(out[2][i] - out[1][i]) for i in range(net.L))
                examples[key] = Example(key, tuple(out[3]), tuple(tuple(int(v) for v in row) for row in occ), float(out[0]), rej, tuple(int(v) for v in out[2]))
    return _finish(settings, net, v0, exact, rev, acc_tot, req_tot, flags, examples, long_len)


def _finish(settings, net, v0, exact, rev, acc_tot, req_tot, flags, examples, long_len):
    p = settings.params
    n = len(rev["dp"])
    dp = np.array(rev["dp"])
    base = float(dp.mean())
    fc = np.array(rev["fcfs"])
    rules = {}
    for r in ANALYSIS_RULES:
        x = np.array(rev[r])
        gap = _stat_of_diff(x - dp, base)
        gap = Stat(100.0 * (float(x.mean()) / base - 1.0), gap.se)
        rules[r] = RuleStat(
            revenue=float(x.mean()),
            gap=gap,
            vs_fcfs=_stat_of_diff(x - fc, base),
            exact_gap=(100.0 * (exact[r] / v0 - 1.0) if r in exact else (0.0 if r == "dp" else None)),
            acc_short=_class_share(acc_tot[r][0], req_tot[0]),
            acc_long=_class_share(acc_tot[r][long_len].sum(), req_tot[long_len].sum()),
            same_as_fcfs=flags[r] == flags["fcfs"],
        )
    sim = _stat_of_diff(dp - v0, v0)
    return Analysis(
        settings=settings, n=n, T=net.T, n_products=len(net.products), states=n_states(net.L, net.cap), v0=v0, sim_vs_v0=sim, rules=rules,
        examples=examples, requested_long=float(req_tot[long_len].sum() / n), requested_short=float(req_tot[0] / n), n_long_products=int(net.long_mask.sum()),
    )


@lru_cache(maxsize=1)
def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Verdikte (reine Funktionen)
def _num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _signed(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}".replace(".", ",").replace("-", "−")


def _pct(x):
    return "—" if x is None else f"{100 * x:.0f} %"


def words(p: Params) -> dict:
    return C.WORDS[p.kulisse]


def verdict_fcfs(a: Analysis) -> tuple:
    """Kernfrage 1: Was kostet die Annahme nach Eingang gegenüber der exakten Rechnung?"""
    p = a.settings.params
    w = words(p)
    f = a.rules["fcfs"]
    d = a.rules["dp"]
    head = (f"Annahme nach Eingang erreicht {_num(100 + f.gap.mean)} % des Ertrags der exakten Rechnung (Mittel über {a.n} Folgen, Standardfehler {_num(f.gap.se)} Prozentpunkte; "
            f"exakter Erwartungswert ohne Simulation: {_num(100 + f.exact_gap)} %).")
    shares = (f" Angenommen werden von den angefragten Fahrten der Länge 1 {_pct(f.acc_short)} und von den {w['long_dat']} {_pct(f.acc_long)}; die exakte Rechnung nimmt {_pct(d.acc_short)} und {_pct(d.acc_long)} an.")
    if f.exact_gap > -1.0:
        return "info", head + shares + " Der Verlust liegt unter einem Prozent."
    return "warning", head + shares


def verdict_rules(a: Analysis) -> tuple:
    """Kernfrage 2: Welche Regel reicht: Bid-Price einmal gerechnet, je Periode neu, oder die exakte Rechnung?"""
    p = a.settings.params
    f, b, r = a.rules["fcfs"], a.rules["bid_static"], a.rules["bid_resolve"]
    text = (f"Gegen die exakte Rechnung (= 100 %) erreichen auf denselben {a.n} Folgen: Annahme nach Eingang {_num(100 + f.gap.mean)} %, Bid-Price einmal gerechnet {_num(100 + b.gap.mean)} %, "
            f"Bid-Price je Periode neu {_num(100 + r.gap.mean)} % (Standardfehler {_num(f.gap.se)}, {_num(b.gap.se)}, {_num(r.gap.se)} Prozentpunkte).")
    kind = "info"
    if b.same_as_fcfs:
        text += f" Der einmal gerechnete Bid-Price nimmt in allen {a.n} Folgen genau dieselben Anfragen an wie die Annahme nach Eingang."
    elif b.vs_fcfs.mean < -2 * b.vs_fcfs.se:
        text += f" Der einmal gerechnete Bid-Price ist hier schlechter als die Annahme nach Eingang ({_signed(b.vs_fcfs.mean)} Prozentpunkte, Standardfehler {_num(b.vs_fcfs.se)})."
        kind = "warning"
    elif b.vs_fcfs.mean > 2 * b.vs_fcfs.se:
        text += f" Der einmal gerechnete Bid-Price holt gegenüber der Annahme nach Eingang {_signed(b.vs_fcfs.mean)} Prozentpunkte (Standardfehler {_num(b.vs_fcfs.se)})."
    else:
        text += f" Zwischen Annahme nach Eingang und einmal gerechnetem Bid-Price ist der Unterschied ({_signed(b.vs_fcfs.mean)} Prozentpunkte) nicht größer als zwei Standardfehler ({_num(b.vs_fcfs.se)})."
    return kind, text


def verdict_forecast(a: Analysis) -> tuple:
    """Kernfrage 3: Wie genau muss die Prognose der Nachfrage nach langen Fahrten sein?"""
    p = a.settings.params
    w = words(p)
    if p.err == 0.0:
        return "info", (f"Die Prognose stimmt hier mit der wahren Nachfrage überein. Mit dem Regler „Prognose für {w['long']}“ lässt sich die erwartete Nachfrage nach {w['long_dat']} um bis zu 50 % verzerren; "
                        "die Referenz (exakte Rechnung mit wahrer Nachfrage) bleibt unverändert.")
    h, b, r, f = a.rules["dp_hat"], a.rules["bid_static"], a.rules["bid_resolve"], a.rules["fcfs"]
    text = (f"Die Regeln rechnen mit {_signed(100 * p.err, 0)} % Nachfrage nach {w['long_dat']}. Auf denselben {a.n} Folgen erreichen gegen die exakte Rechnung mit wahrer Nachfrage: exakte Rechnung mit der falschen Prognose "
            f"{_num(100 + h.gap.mean)} % (exakter Erwartungswert {_num(100 + h.exact_gap)} %), Bid-Price einmal gerechnet {_num(100 + b.gap.mean)} %, je Periode neu {_num(100 + r.gap.mean)} %; "
            f"Annahme nach Eingang (braucht keine Prognose) {_num(100 + f.gap.mean)} %.")
    return ("warning" if h.exact_gap < -2.0 else "info"), text


# ---------------------------------------------------------------- Zugriff auf die Messreihen
def fmt_pct(x, digits=1):
    return _num(x, digits)


def raster_summary(rows) -> dict:
    """Zählungen über das ganze vorgerechnete Raster (Experimente 1 bis 3); jede Aussage der README und der App über das Raster stammt von hier."""
    over = [r for r in rows if r["rho"] >= 2.0 and r["alpha"] <= 0.6]
    low = [r for r in rows if r["rho"] == 0.8]
    a1 = [r for r in rows if r["alpha"] == 1.0]
    worse = [r for r in rows if r["bid_vs_fcfs_se"] > 0 and r["bid_vs_fcfs"] < -2 * r["bid_vs_fcfs_se"]]
    better = [r for r in rows if r["bid_vs_fcfs_se"] > 0 and r["bid_vs_fcfs"] > 2 * r["bid_vs_fcfs_se"]]
    return {
        "cells": len(rows),
        "fcfs_min": min(r["fcfs"]["gap"] for r in rows), "fcfs_max": max(r["fcfs"]["gap"] for r in rows),
        "bid_min": min(r["bid_static"]["gap"] for r in rows), "bid_max": max(r["bid_static"]["gap"] for r in rows),
        "low_load_fcfs_min": min(r["fcfs"]["gap"] for r in low),
        "over_cells": len(over),
        "over_fcfs_min": min(r["fcfs"]["gap"] for r in over), "over_fcfs_max": max(r["fcfs"]["gap"] for r in over),
        "over_bid_min": min(r["bid_static"]["gap"] for r in over), "over_bid_max": max(r["bid_static"]["gap"] for r in over),
        "a1_cells": len(a1), "a1_same": sum(1 for r in a1 if r["bid_same_as_fcfs"]),
        "a1_fcfs_min": min(r["fcfs"]["gap"] for r in a1), "a1_fcfs_max": max(r["fcfs"]["gap"] for r in a1),
        "same_cells": sum(1 for r in rows if r["bid_same_as_fcfs"]),
        "worse": worse, "better": len(better),
        "worse_exact": sum(1 for r in rows if r["bid_static"]["exact_gap"] < r["fcfs"]["exact_gap"] - 1e-9),        # nach dem exakten Erwartungswert beider Regeln
        "bid_vs_fcfs_max": max(r["bid_vs_fcfs"] for r in rows),
    }


def forecast_summary(rows) -> dict:
    """Zählungen über die Prognosemessung (Kernfrage 3, Experiment der README)."""
    return {
        "rows": len(rows),
        "dp_hat_worst": max(-r["dp_hat"]["exact_gap"] for r in rows),
        "dp_hat_worst_sim": max(-r["dp_hat"]["gap"] for r in rows),
        "fcfs_min": min(r["fcfs"]["gap"] for r in rows), "fcfs_max": max(r["fcfs"]["gap"] for r in rows),
    }


def alpha_one_resolve(rows) -> dict:
    """Prognose-Szenario D (gleicher Preis je Teilstrecke, Last 1,2): Bereich der Annahme nach Eingang und des Bid-Price je Periode neu über alle Verzerrungsstufen."""
    d = [r for r in rows if r["szenario"] == "D"]
    return {
        "rows": len(d),
        "fcfs_min": min(r["fcfs"]["gap"] for r in d), "fcfs_max": max(r["fcfs"]["gap"] for r in d),
        "bid_resolve_min": min(r["bid_resolve"]["gap"] for r in d), "bid_resolve_max": max(r["bid_resolve"]["gap"] for r in d),
        "resolve_beats_fcfs": all(r["bid_resolve"]["vs_fcfs"] > 2 * r["bid_resolve"]["vs_fcfs_se"] for r in d),
    }
