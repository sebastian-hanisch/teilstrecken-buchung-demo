"""Erzeugt `data/tsb_results.json`: vorgerechnete Messreihen (Anfragefolgen mit festen Zufalls-Seeds, exakte Rechnung als Maßstab).

    python tools/make_results.py <teil> [<teil> ...]      Teile: raster hotel prognose netz schutz zustaende

Jeder Teil schreibt seinen Abschnitt und lässt die übrigen stehen. Jede Zelle hat ihren eigenen Zufalls-Seed (Basis des Teils + Zellennummer), Folge k einer Zelle hat den Zufalls-Seed `Zellseed · 1000 + k`.
`raster` und `hotel` rechnen ohne die Neuberechnung des Bid-Price (zu teuer für 1000 Folgen je Zelle), `prognose` und `netz` mit; `schutz` wählt die Schutzstufe k auf Trainingsfolgen und misst auf frischen Folgen."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tsb_algorithm as A  # noqa: E402
import tsb_constants as C  # noqa: E402
import tsb_evaluation as E  # noqa: E402
from tsb_rng import SplitMix64  # noqa: E402
from tsb_scenario import Params, build_net, n_states, sample_sequence  # noqa: E402

OUT = ROOT / "data" / "tsb_results.json"
N_FAST = 1000        # Folgen je Zelle ohne Neuberechnung
N_RESOLVE = 500      # Folgen je Zelle mit Neuberechnung
BASE = {"raster": 100000, "hotel": 500000, "prognose": 200000, "netz": 300000, "schutz": 400000}
SCENARIOS = {        # Szenarien der Prognose- und Schutzmessung (Netz mit 4 Teilstrecken, Kapazität 8)
    "A": ("Überlast, lange Fahrten billig und früh", dict(rho=2.0, alpha=0.3, early=1.0)),
    "B": ("Last 1,2, lange Fahrten billig", dict(rho=1.2, alpha=0.6, early=0.0)),
    "C": ("starke Überlast, lange Fahrten billig", dict(rho=3.0, alpha=0.6, early=1.0)),
    "D": ("gleicher Preis je Teilstrecke", dict(rho=1.2, alpha=1.0, early=0.0)),
}


def load() -> dict:
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def save(data: dict):
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes((json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))   # LF auch unter Windows


def fast_cell(params: Params, cell_seed: int, n: int) -> dict:
    """Annahme nach Eingang, Bid-Price (einmal gerechnet) und exakte Rechnung auf n Folgen; dazu die exakten Erwartungswerte."""
    net = build_net(params)
    ctx = A.Context(net, net.lam)
    v0 = float(ctx.V[0][(net.cap,) * net.L])
    exact = {"fcfs": A.value_of_policy(net, "fcfs"), "bid_static": A.value_of_policy(net, "bid_static", pi=ctx.pi_static)}
    rev = {r: [] for r in ("fcfs", "bid_static", "dp")}
    same = True
    for k in range(n):
        seq = sample_sequence(net, SplitMix64(cell_seed * C.SUBSEED_STRIDE + k))
        out = {r: A.run_policy(net, seq, r, ctx) for r in rev}
        for r in rev:
            rev[r].append(out[r][0])
        same &= tuple(e[2] for e in out["fcfs"][3]) == tuple(e[2] for e in out["bid_static"][3])
    dp = np.array(rev["dp"])
    base = float(dp.mean())
    row = {"T": net.T, "n": n, "v0": v0, "dp": base, "sim_vs_v0": E._stat_of_diff(dp - v0, v0).mean, "sim_vs_v0_se": E._stat_of_diff(dp - v0, v0).se, "bid_same_as_fcfs": bool(same)}
    for r in ("fcfs", "bid_static"):
        x = np.array(rev[r])
        row[r] = {"gap": 100.0 * (float(x.mean()) / base - 1.0), "se": E._stat_of_diff(x - dp, base).se, "exact_gap": 100.0 * (exact[r] / v0 - 1.0)}
    d = E._stat_of_diff(np.array(rev["bid_static"]) - np.array(rev["fcfs"]), base)
    row["bid_vs_fcfs"], row["bid_vs_fcfs_se"] = d.mean, d.se
    return row


def part_raster(data):
    rows = []
    cell = 0
    for cap, rho, alpha, early in itertools.product((4, 8, 16), (0.8, 1.2, 2.0, 3.0), (1.0, 0.6, 0.3), (0.0, 1.0)):
        p = Params(L=4, cap=cap, rho=rho, alpha=alpha, early=early)
        row = {"cap": cap, "rho": rho, "alpha": alpha, "early": early, **fast_cell(p, BASE["raster"] + cell, N_FAST)}
        rows.append(row)
        cell += 1
        print("raster", cell, cap, rho, alpha, early, round(row["fcfs"]["gap"], 1), round(row["bid_static"]["gap"], 1), flush=True)
    data["raster"] = rows


def part_hotel(data):
    rows = []
    cell = 0
    for rho, alpha in itertools.product((1.2, 2.0, 3.0), (1.0, 0.6, 0.3)):
        p = Params(kulisse="hotel", L=5, cap=8, rho=rho, alpha=alpha, early=1.0)
        row = {"rho": rho, "alpha": alpha, **fast_cell(p, BASE["hotel"] + cell, N_FAST)}
        rows.append(row)
        cell += 1
        print("hotel", cell, rho, alpha, round(row["fcfs"]["gap"], 1), round(row["bid_static"]["gap"], 1), flush=True)
    data["hotel"] = rows


def analysis_row(a: E.Analysis) -> dict:
    out = {"n": a.n, "T": a.T, "v0": a.v0, "sim_vs_v0": a.sim_vs_v0.mean, "sim_vs_v0_se": a.sim_vs_v0.se}
    for r, s in a.rules.items():
        out[r] = {"gap": s.gap.mean, "se": s.gap.se, "exact_gap": s.exact_gap, "vs_fcfs": s.vs_fcfs.mean, "vs_fcfs_se": s.vs_fcfs.se, "acc_short": s.acc_short, "acc_long": s.acc_long}
    return out


def part_prognose(data):
    rows = []
    cell = 0
    for key, (label, kw) in SCENARIOS.items():
        for err in C.ERR_STEPS:
            p = Params(L=4, cap=8, err=err, **kw)
            a = E.analyse(E.Settings(params=p, seed=BASE["prognose"] + cell), n_seq=N_RESOLVE)
            rows.append({"szenario": key, "label": label, "err": err, **analysis_row(a)})
            cell += 1
            print("prognose", key, err, round(a.rules["dp_hat"].gap.mean, 2), round(a.rules["bid_resolve"].gap.mean, 2), flush=True)
    data["prognose"] = rows


def part_netz(data):
    rows = []
    for cell, L in enumerate(range(2, 7)):
        p = Params(L=L, cap=6, rho=2.0, alpha=0.3, early=1.0)
        a = E.analyse(E.Settings(params=p, seed=BASE["netz"] + cell), n_seq=N_RESOLVE)
        rows.append({"L": L, "cap": 6, "zustaende": n_states(L, 6), "produkte": a.n_products, **analysis_row(a)})
        print("netz", L, round(a.rules["fcfs"].gap.mean, 1), round(a.rules["bid_resolve"].gap.mean, 1), flush=True)
    data["netz"] = rows


def protect_revenue(net, seq, k):
    """Fester Schutz: eine lange Fahrt (mehr als die halbe Linie) wird nur angenommen, wenn danach auf jeder ihrer Teilstrecken noch k Plätze frei bleiben; alles andere nach Eingang."""
    x = [net.cap] * net.L
    rev = 0.0
    long_mask = net.long_mask
    for p in seq:
        if p < 0:
            continue
        i, j = net.products[p]
        need = 1 + (k if long_mask[p] else 0)
        if any(x[l] < need for l in range(i, j)):
            continue
        rev += net.fare[p]
        for l in range(i, j):
            x[l] -= 1
    return rev


def part_schutz(data):
    rows = []
    for cell, (key, (label, kw)) in enumerate(SCENARIOS.items()):
        p = Params(L=4, cap=8, **kw)
        net = build_net(p)
        train = [sample_sequence(net, SplitMix64((BASE["schutz"] + cell) * C.SUBSEED_STRIDE + k)) for k in range(500)]
        scores = [float(np.mean([protect_revenue(net, s, k) for s in train])) for k in range(0, net.cap + 1)]
        best_k = int(np.argmax(scores))                      # kleinste Stufe unter den besten (argmax nimmt die erste); k = Kapazität heißt: keine lange Fahrt annehmen
        ctx = A.Context(net, net.lam)
        test = [sample_sequence(net, SplitMix64((BASE["schutz"] + 50 + cell) * C.SUBSEED_STRIDE + k)) for k in range(N_FAST)]
        dp = np.array([A.run_policy(net, s, "dp", ctx)[0] for s in test])
        base = float(dp.mean())
        row = {"szenario": key, "label": label, "n_train": len(train), "n_test": len(test), "best_k": best_k, "T": net.T}
        for name, vals in (("fcfs", [A.run_policy(net, s, "fcfs", ctx)[0] for s in test]), ("bid_static", [A.run_policy(net, s, "bid_static", ctx)[0] for s in test]),
                           ("schutz", [protect_revenue(net, s, best_k) for s in test])):
            x = np.array(vals)
            row[name] = {"gap": 100.0 * (float(x.mean()) / base - 1.0), "se": E._stat_of_diff(x - dp, base).se}
        rows.append(row)
        print("schutz", key, best_k, round(row["fcfs"]["gap"], 1), round(row["schutz"]["gap"], 1), round(row["bid_static"]["gap"], 1), flush=True)
    data["schutz"] = rows


def part_zustaende(data):
    rows = []
    for L, cap in ((4, 8), (4, 16), (5, 8), (5, 9), (6, 6), (6, 8), (6, 10)):
        net = build_net(Params(L=L, cap=cap))
        s = n_states(L, cap)
        rows.append({"L": L, "cap": cap, "zustaende": s, "T": net.T, "mb": (net.T + 1) * s * 8 / 1e6, "live": s <= C.STATES_LIVE_MAX})
    data["zustaende"] = rows


PARTS = {"raster": part_raster, "hotel": part_hotel, "prognose": part_prognose, "netz": part_netz, "schutz": part_schutz, "zustaende": part_zustaende}


def main(argv):
    names = argv or list(PARTS)
    for n in names:
        if n not in PARTS:
            raise SystemExit(f"unbekannter Teil {n}; erlaubt: {', '.join(PARTS)}")
    for n in names:
        data = load()                       # frisch lesen: parallel laufende Teile schreiben nur ihren eigenen Abschnitt
        PARTS[n](data)
        fresh = load()
        fresh[n] = data[n]
        save(fresh)
        print("geschrieben:", n)


if __name__ == "__main__":
    main(sys.argv[1:])
