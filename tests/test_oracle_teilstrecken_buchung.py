"""Orakel-Tests (ORACLE-PLAYBOOK): die Kernroutinen gegen unabhängige Rechenwege.

* Exakte Rechnung (`dp_values`, Zustandsarrays mit Schnitten) gegen eine Baumrekursion in reiner Python-Rechnung mit Brüchen (Erwartungswert über alle Ankünfte, Entscheidung je Knoten)
  und gegen die Aufzählung ALLER Regeln, die vom Zustand abhängen, bewertet durch Aufzählung aller Anfragefolgen (kein Bellman-Argument im Orakel).
* Exakter Erwartungswert einer Regel (`value_of_policy`) gegen eine Rekursion mit Wörterbuch; Simulation (`run_policy`) gegen diesen exakten Wert.
* Bid-Price: die Schattenpreise sind Duale des Optimierungsproblems, geprüft über die starke Dualität (zweites LP, das Dualproblem selbst).
* Annahme nach Eingang gegen eine zweite, einfache Umsetzung.
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import linprog

import tsb_algorithm as A
import tsb_constants as C
from tsb_rng import SplitMix64
from tsb_scenario import Net, Params, build_net, sample_sequence


# ---------------------------------------------------------------- unabhängiges Orakel
def products(L):
    return [(i, j) for i in range(L) for j in range(i + 1, L + 1)]


def fits(x, prod):
    i, j = prod
    return all(x[l] >= 1 for l in range(i, j))


def take(x, prod):
    i, j = prod
    return tuple(x[l] - 1 if i <= l < j else x[l] for l in range(len(x)))


def expectimax(L, cap, fares, lam, t=0, x=None):
    """Baumrekursion über Ankunft (Zufallsknoten) und Entscheidung (Maximumknoten), exakt in Brüchen; `lam[t][p]` sind Brüche."""
    prods = products(L)
    x = x if x is not None else (cap,) * L
    if t == len(lam):
        return Fraction(0)
    total = Fraction(0)
    p_none = 1 - sum(lam[t])
    total += p_none * expectimax(L, cap, fares, lam, t + 1, x)
    for p, prod in enumerate(prods):
        reject = expectimax(L, cap, fares, lam, t + 1, x)
        best = reject
        if fits(x, prod):
            best = max(reject, fares[p] + expectimax(L, cap, fares, lam, t + 1, take(x, prod)))
        total += lam[t][p] * best
    return total


def all_markov_rules_best(L, cap, fares, lam):
    """Beste Regel unter ALLEN Regeln „(Periode, Zustand, Produkt) -> annehmen oder ablehnen“, bewertet durch Aufzählung aller Anfragefolgen."""
    prods = products(L)
    T = len(lam)
    points = []                                    # alle Entscheidungspunkte, an denen die Wahl überhaupt eine ist (das Produkt passt und der Zustand ist erreichbar)
    reach = {(cap,) * L}
    for t in range(T):
        for x in sorted(reach):
            for p, prod in enumerate(prods):
                if fits(x, prod):
                    points.append((t, x, p))
        reach = reach | {take(x, prod) for x in reach for prod in prods if fits(x, prod)}
    assert len(points) <= 16, "Aufzählung zu groß"
    best = Fraction(-1)
    seqs = list(itertools.product(range(-1, len(prods)), repeat=T))
    prob = {}
    for seq in seqs:
        pr = Fraction(1)
        for t, p in enumerate(seq):
            pr *= (1 - sum(lam[t])) if p < 0 else lam[t][p]
        prob[seq] = pr
    for bits in itertools.product((False, True), repeat=len(points)):
        accept = dict(zip(points, bits))
        value = Fraction(0)
        for seq in seqs:
            x = (cap,) * L
            rev = Fraction(0)
            for t, p in enumerate(seq):
                if p >= 0 and fits(x, prods[p]) and accept[(t, x, p)]:
                    rev += fares[p]
                    x = take(x, prods[p])
            value += prob[seq] * rev
        best = max(best, value)
    return best


def policy_value_dict(L, cap, fares, lam, decide):
    """Exakter Erwartungswert einer zustandsabhängigen Regel `decide(t, x, p)` durch Rekursion mit Wörterbuch (ohne Arrays)."""
    prods = products(L)
    T = len(lam)
    memo = {}

    def rec(t, x):
        if t == T:
            return 0.0
        if (t, x) in memo:
            return memo[(t, x)]
        v = (1.0 - sum(lam[t])) * rec(t + 1, x)
        for p, prod in enumerate(prods):
            if fits(x, prod) and decide(t, x, p):
                v += lam[t][p] * (fares[p] + rec(t + 1, take(x, prod)))
            else:
                v += lam[t][p] * rec(t + 1, x)
        memo[(t, x)] = v
        return v

    return rec(0, (cap,) * L)


def fcfs_simulation_plain(net, seq):
    """Annahme nach Eingang als einfachste Umsetzung: Liste freier Plätze, jede passende Anfrage nehmen."""
    free = [net.cap] * net.L
    rev = 0.0
    for p in seq:
        if p < 0:
            continue
        i, j = net.products[p]
        if min(free[i:j]) >= 1:
            rev += net.fare[p]
            for l in range(i, j):
                free[l] -= 1
    return rev


# ---------------------------------------------------------------- Tests
def _fraction_instance(L, cap, T, num):
    prods = products(L)
    rng = SplitMix64(num)
    fares = [Fraction(int(10 * (j - i) ** 0.7 * 2)) / 2 + Fraction(rng.below(5)) for (i, j) in prods]
    lam = []
    for _ in range(T):
        w = [rng.below(4) for _ in prods]
        s = sum(w) + 3
        lam.append([Fraction(v, s + 2) for v in w])               # Summe < 1
    return fares, lam


@pytest.mark.parametrize("L,cap,T,num", [(1, 1, 3, 1), (2, 1, 3, 2), (2, 1, 4, 3), (2, 2, 3, 4), (3, 1, 3, 5), (2, 2, 4, 6)])
def test_exact_rule_equals_the_expectimax_tree_with_fractions(L, cap, T, num):
    fares, lam = _fraction_instance(L, cap, T, num)
    net = Net.from_arrays(L, cap, [float(f) for f in fares], [[float(lam[t][p]) for t in range(T)] for p in range(len(fares))])
    V = A.dp_values(net, net.lam)
    assert float(V[0][(cap,) * L]) == pytest.approx(float(expectimax(L, cap, fares, lam)), rel=1e-12)


@pytest.mark.parametrize("num", [11, 12, 13])
def test_exact_rule_equals_the_best_of_all_state_dependent_rules(num):
    L, cap, T = 2, 1, 3
    fares, lam = _fraction_instance(L, cap, T, num)
    net = Net.from_arrays(L, cap, [float(f) for f in fares], [[float(lam[t][p]) for t in range(T)] for p in range(len(fares))])
    V = A.dp_values(net, net.lam)
    best = all_markov_rules_best(L, cap, fares, lam)
    assert float(V[0][(cap,) * L]) == pytest.approx(float(best), rel=1e-12)


def test_exact_value_of_the_rules_equals_the_dictionary_recursion():
    net = build_net(Params(L=3, cap=2, rho=2.0, alpha=0.3))
    ctx = A.Context(net, net.lam)
    fares = [float(f) for f in net.fare]
    lam = [[float(net.lam[p, t]) for p in range(len(fares))] for t in range(net.T)]
    prods = net.products
    assert A.value_of_policy(net, "fcfs") == pytest.approx(policy_value_dict(3, 2, fares, lam, lambda t, x, p: True), rel=1e-10)
    pi = ctx.pi_static
    dec_bid = lambda t, x, p: fares[p] >= sum(pi[prods[p][0]:prods[p][1]]) - 1e-12
    assert A.value_of_policy(net, "bid_static", pi=pi) == pytest.approx(policy_value_dict(3, 2, fares, lam, dec_bid), rel=1e-10)
    # exakte Rechnung: Entscheidung aus einer zweiten Wertfunktion (Wörterbuch-Rekursion, ohne die Arrays der App)
    memo = {}

    def val(t, x):
        if t == net.T:
            return 0.0
        if (t, x) in memo:
            return memo[(t, x)]
        v = (1.0 - sum(lam[t])) * val(t + 1, x)
        for p, prod in enumerate(prods):
            rej = val(t + 1, x)
            v += lam[t][p] * (max(rej, fares[p] + val(t + 1, take(x, prod))) if fits(x, prod) else rej)
        memo[(t, x)] = v
        return v

    dec_dp = lambda t, x, p: fares[p] + val(t + 1, take(x, prods[p])) >= val(t + 1, x) - 1e-12
    assert A.value_of_policy(net, "dp", V_hat=ctx.V) == pytest.approx(policy_value_dict(3, 2, fares, lam, dec_dp), rel=1e-10)
    assert val(0, (2, 2, 2)) == pytest.approx(float(ctx.V[0][2, 2, 2]), rel=1e-10)


@pytest.mark.parametrize("params,seed", [(Params(L=3, cap=3, rho=2.0, alpha=0.3), 1), (Params(L=3, cap=3, rho=1.2, alpha=0.6, early=0.0), 2), (Params(L=2, cap=4, rho=2.0, alpha=1.0, early=0.0), 3)])
def test_simulation_matches_the_exact_expected_value(params, seed):
    net = build_net(params)
    ctx = A.Context(net, net.lam)
    exact = {"fcfs": A.value_of_policy(net, "fcfs"), "bid_static": A.value_of_policy(net, "bid_static", pi=ctx.pi_static), "dp": A.value_of_policy(net, "dp", V_hat=ctx.V)}
    n = 1500
    revs = {r: [] for r in exact}
    for k in range(n):
        seq = sample_sequence(net, SplitMix64(seed * 100000 + k))
        for r in exact:
            revs[r].append(A.run_policy(net, seq, r, ctx)[0])
    for r, vals in revs.items():
        v = np.array(vals)
        se = v.std(ddof=1) / np.sqrt(n)
        assert abs(v.mean() - exact[r]) < 4.5 * se, (r, v.mean(), exact[r], se)


@pytest.mark.parametrize("num", range(8))
def test_first_come_first_served_equals_the_plain_implementation(num):
    rng = SplitMix64(900 + num)
    params = Params(L=2 + rng.below(4), cap=2 + rng.below(4), rho=C.RHO_STEPS[rng.below(len(C.RHO_STEPS))], alpha=C.ALPHA_STEPS[rng.below(len(C.ALPHA_STEPS))], early=C.EARLY_STEPS[rng.below(3)])
    net = build_net(params)
    ctx = A.Context(net, net.lam, need_dp=False)
    for k in range(10):
        seq = sample_sequence(net, SplitMix64(num * 100 + k))
        assert A.run_policy(net, seq, "fcfs", ctx)[0] == pytest.approx(fcfs_simulation_plain(net, seq))


@pytest.mark.parametrize("params", [Params(), Params(L=3, cap=3, rho=3.0, alpha=0.6), Params(L=5, cap=3, rho=1.5, alpha=0.3, kulisse="hotel"), Params(L=2, cap=6, rho=2.0, alpha=1.0, early=0.0)])
def test_bid_prices_are_dual_optimal_by_strong_duality(params):
    """Das Dualproblem min K·π + D·z unter Aᵀπ + z ≥ f, π, z ≥ 0 wird eigens gelöst; sein Wert gleicht dem des Primalproblems, und π aus dem Kern ist dafür zulässig und erreicht ihn."""
    net = build_net(params)
    K = np.full(net.L, float(net.cap))
    primal = linprog(-net.fare, A_ub=net.A, b_ub=K, bounds=list(zip(np.zeros(len(net.fare)), net.D)), method="highs")
    dual = linprog(np.concatenate([K, net.D]), A_ub=-np.hstack([net.A.T, np.eye(len(net.fare))]), b_ub=-net.fare, bounds=[(0, None)] * (net.L + len(net.fare)), method="highs")
    assert primal.status == 0 and dual.status == 0
    assert -primal.fun == pytest.approx(dual.fun, rel=1e-9)
    pi = A.bid_prices(net, K, net.D)
    assert (pi >= -1e-9).all()
    z = np.maximum(net.fare - net.A.T @ pi, 0.0)                    # kleinstes z, das die Dualbedingung erfüllt
    assert K @ pi + net.D @ z == pytest.approx(-primal.fun, rel=1e-7)


def test_neuberechnung_at_the_start_equals_the_static_bid_price():
    for params in (Params(), Params(L=3, cap=3, rho=3.0, alpha=0.6), Params(rho=0.8, alpha=0.6, early=0.0)):
        net = build_net(params)
        ctx = A.Context(net, net.lam, need_dp=False)
        full = [net.cap] * net.L
        for p in range(len(net.fare)):
            assert A.accepts(ctx, "bid_resolve", 0, full, p) == A.accepts(ctx, "bid_static", 0, full, p)
