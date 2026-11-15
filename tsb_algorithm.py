"""Kern: exakte Rechnung (DP über den Kapazitätsvektor), Bid-Price (Duale des deterministischen LP) und die vier Annahmeregeln.

Zustand x = freie Plätze je Teilstrecke, ein Vektor mit L Einträgen von 0 bis cap. Die DP rechnet rückwärts

    V[t][x] = V[t+1][x] + Σ_p lam[p, t] · max(0, fare_p + V[t+1][x - a_p] - V[t+1][x])     (nur wo x - a_p >= 0),

V[T] = 0, und nimmt die Anfrage p im Zustand x in Periode t an, wenn fare_p + V[t+1][x - a_p] >= V[t+1][x] und Platz da ist (a_p = Inzidenzvektor des Produkts).

Regeln (alle nehmen nie an, was nicht passt):
  fcfs         annehmen, solange auf allen Teilstrecken Platz ist
  bid_static   annehmen, wenn der Preis mindestens die Summe der Schattenpreise der belegten Teilstrecken deckt; Schattenpreise = Duale des LP
               max fare·y  unter  A y <= cap,  0 <= y <= Nachfrage  (einmal zu Beginn)
  bid_resolve  wie bid_static, aber das LP wird in jeder Periode mit der Restkapazität und der Restnachfrage neu gelöst
  dp           die exakte Rechnung oben
"""

import numpy as np
from scipy.optimize import linprog

TOL = 1e-12


def _slices(net):
    """Je Produkt zwei Schnitte durch das Zustandsarray: sx = Zustände, in denen das Produkt passt, se = die Zustände danach."""
    L = net.L
    out = []
    for (i, j) in net.products:
        sx = tuple(slice(1, None) if i <= l < j else slice(None) for l in range(L))
        se = tuple(slice(0, -1) if i <= l < j else slice(None) for l in range(L))
        out.append((sx, se))
    return out


def dp_values(net, lam):
    """Wertfunktion V[t] (Arrays der Form (cap + 1,) * L) für die Nachfrage `lam`, t = 0 bis T; V[T] = 0."""
    shape = (net.cap + 1,) * net.L
    sl = _slices(net)
    V = [None] * (net.T + 1)
    V[net.T] = np.zeros(shape)
    for t in range(net.T - 1, -1, -1):
        Vn = V[t + 1]
        Vt = Vn.copy()
        for p in range(len(net.fare)):
            sx, se = sl[p]
            gain = np.zeros(shape)
            gain[sx] = np.maximum(net.fare[p] + Vn[se] - Vn[sx], 0.0)
            Vt += lam[p, t] * gain
        V[t] = Vt
    return V


def value_of_policy(net, rule, V_hat=None, pi=None):
    """Exakter Erwartungswert des Ertrags einer zustandsabhängigen Regel unter der WAHREN Nachfrage `net.lam`, ohne Simulation.

    rule = "fcfs": jede Anfrage annehmen, die passt; "bid_static": annehmen, wenn fare >= pi[i:j].sum() (pi: Schattenpreise);
    "dp": annehmen, wenn fare + V_hat[t+1][x - a] >= V_hat[t+1][x] (V_hat: Wertfunktion für eine Prognose, die von der wahren abweichen darf)."""
    shape = (net.cap + 1,) * net.L
    sl = _slices(net)
    W = np.zeros(shape)
    for t in range(net.T - 1, -1, -1):
        Wt = W.copy()
        for p, (i, j) in enumerate(net.products):
            sx, se = sl[p]
            gain_here = net.fare[p] + W[se] - W[sx]
            if rule == "fcfs":
                take = True
            elif rule == "bid_static":
                take = bool(net.fare[p] >= pi[i:j].sum() - TOL)
            elif rule == "dp":
                Vn = V_hat[t + 1]
                take = net.fare[p] + Vn[se] >= Vn[sx] - TOL
            else:
                raise ValueError(rule)
            gain = np.zeros(shape)
            gain[sx] = np.where(take, gain_here, 0.0)
            Wt += net.lam[p, t] * gain
        W = Wt
    return float(W[(net.cap,) * net.L])


def bid_prices(net, cap_vec, demand):
    """Schattenpreise der Teilstrecken: Duale der Kapazitätsrestriktionen des LP max fare·y, A y <= cap_vec, 0 <= y <= demand."""
    res = linprog(-net.fare, A_ub=net.A, b_ub=np.asarray(cap_vec, dtype=float), bounds=list(zip(np.zeros(len(net.fare)), demand)), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP für die Schattenpreise nicht gelöst (Status {res.status}: {res.message})")   # nie still Nullpreise liefern: das wäre die Annahme nach Eingang
    return -res.ineqlin.marginals


class Context:
    """Alles, was die Regeln über die Nachfrage WISSEN: die Prognose lam_hat (kann von der wahren Nachfrage abweichen) und was daraus folgt."""

    def __init__(self, net, lam_hat, need_dp=True):
        self.net = net
        self.lam_hat = lam_hat
        self.demand_hat = lam_hat.sum(axis=1)
        self.rem_hat = np.cumsum(lam_hat[:, ::-1], axis=1)[:, ::-1]      # rem_hat[:, t] = Prognose der Nachfrage der Perioden t bis T-1
        self.pi_static = bid_prices(net, np.full(net.L, float(net.cap)), self.demand_hat)
        self.V = dp_values(net, lam_hat) if need_dp else None


def accepts(ctx, rule, t, x, p):
    """Nimmt die Regel die Anfrage p in Periode t im Zustand x (Liste freier Plätze) an? Passt sie nicht, ist die Antwort immer nein."""
    net = ctx.net
    i, j = net.products[p]
    if any(x[l] < 1 for l in range(i, j)):
        return False
    if rule == "fcfs":
        return True
    if rule == "bid_static":
        return bool(net.fare[p] >= ctx.pi_static[i:j].sum() - TOL)
    if rule == "bid_resolve":
        pi = bid_prices(net, np.array(x, dtype=float), np.maximum(ctx.rem_hat[:, t], 0.0))
        return bool(net.fare[p] >= pi[i:j].sum() - TOL)
    if rule == "dp":
        y = list(x)
        for l in range(i, j):
            y[l] -= 1
        Vn = ctx.V[t + 1]
        return bool(net.fare[p] + Vn[tuple(y)] >= Vn[tuple(x)] - TOL)
    raise ValueError(rule)


def run_policy(net, seq, rule, ctx):
    """Eine Anfragefolge unter einer Regel. Rückgabe: (Ertrag, angenommen je Länge, angefragt je Länge, Ereignisse); Ereignis = (t, Produkt, angenommen?)."""
    x = [net.cap] * net.L
    rev = 0.0
    req = np.zeros(net.L, dtype=int)
    acc = np.zeros(net.L, dtype=int)
    events = []
    for t, p in enumerate(seq):
        if p < 0:
            continue
        i, j = net.products[p]
        req[j - i - 1] += 1
        take = accepts(ctx, rule, t, x, p)
        events.append((t, p, take))
        if take:
            rev += net.fare[p]
            acc[j - i - 1] += 1
            for l in range(i, j):
                x[l] -= 1
    return rev, acc, req, events


def occupancy_by_span(net, events):
    """Belegte Plätze je Teilstrecke, aufgeteilt nach Länge der Fahrt: Array (Teilstrecken, Länge 1 bis L)."""
    occ = np.zeros((net.L, net.L), dtype=int)
    for (_t, p, take) in events:
        if take:
            i, j = net.products[p]
            for l in range(i, j):
                occ[l, j - i - 1] += 1
    return occ
