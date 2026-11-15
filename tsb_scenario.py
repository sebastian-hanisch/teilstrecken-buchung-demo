"""Szenario: Netz aus Teilstrecken, Produkte, Preise, Ankunftswahrscheinlichkeiten je Periode und der Folgenerzeuger.

Eine Linie mit L Teilstrecken (Stationen 0 bis L). Ein Produkt (i, j) mit 0 <= i < j <= L belegt je einen Platz auf den Teilstrecken i bis j-1 und zahlt
10 · (j - i) ** alpha. In jeder der T Perioden kommt höchstens eine Anfrage: Produkt p mit der Wahrscheinlichkeit lam[p, t]."""

import math
from dataclasses import dataclass

import numpy as np

import tsb_constants as C
from tsb_rng import SplitMix64


@dataclass(frozen=True, kw_only=True)
class Params:
    kulisse: str = C.DEFAULT_KULISSE
    L: int = C.DEFAULT_L
    cap: int = C.DEFAULT_CAP
    rho: float = C.DEFAULT_RHO
    alpha: float = C.DEFAULT_ALPHA
    early: float = C.DEFAULT_EARLY
    err: float = C.DEFAULT_ERR


def cap_max(L: int) -> int:
    """Größte Kapazität je Teilstrecke, für die die exakte Rechnung bei L Teilstrecken live noch höchstens STATES_LIVE_MAX Zustände hat."""
    c = C.CAP_MIN
    while c + 1 <= C.CAP_MAX and (c + 2) ** L <= C.STATES_LIVE_MAX:
        c += 1
    return c


def n_states(L: int, cap: int) -> int:
    return (cap + 1) ** L


class Net:
    """Ein Netz mit Nachfrage. `lam` hat die Form (Produkte, Perioden); `span[p]` ist die Länge des Produkts p."""

    def __init__(self, L, cap, rho, alpha, early, gamma, fare_base=C.FARE_BASE, load_per_period=C.LOAD_PER_PERIOD):
        self.L, self.cap = L, cap
        self.products = [(i, j) for i in range(L) for j in range(i + 1, L + 1)]
        P = len(self.products)
        self.A = np.zeros((L, P))
        for k, (i, j) in enumerate(self.products):
            self.A[i:j, k] = 1.0
        self.span = np.array([j - i for (i, j) in self.products])
        self.fare = np.array([fare_base * float(s) ** alpha for s in self.span])
        w = np.exp(-gamma * (self.span - 1.0))
        load = self.A @ w                                       # Nachfrage je Teilstrecke bei Gewicht 1 je Produkt
        D = w * (rho * cap) / load.mean()                       # erwartete Gesamtnachfrage je Produkt: mittlere Last der Teilstrecken = rho · cap
        self.T = int(math.ceil(D.sum() / load_per_period))
        t = (np.arange(self.T) + 0.5) / self.T
        sgn = (self.span - self.span.mean()) / (self.span.max() - self.span.mean() + 1e-9)       # lange Fahrten > 0, kurze < 0
        shape = np.clip(1.0 + early * sgn[:, None] * (1.0 - 2.0 * t[None, :]), 0.05, None)      # early > 0: lange früh, kurze spät
        shape = shape / shape.mean(axis=1, keepdims=True)
        lam = D[:, None] / self.T * shape
        s = lam.sum(axis=0)
        if s.max() > C.MAX_ARRIVAL:
            lam = lam / s.max() * C.MAX_ARRIVAL
        self.lam = lam
        self.D = lam.sum(axis=1)
        self.cum = np.cumsum(lam, axis=0)                       # für den Folgenerzeuger

    @classmethod
    def from_arrays(cls, L, cap, fare, lam):
        """Netz mit vorgegebenen Preisen und Ankunftswahrscheinlichkeiten (Mini-Instanzen der Tests); `fare` je Produkt, `lam` (Produkte, Perioden)."""
        net = cls.__new__(cls)
        net.L, net.cap = L, cap
        net.products = [(i, j) for i in range(L) for j in range(i + 1, L + 1)]
        net.A = np.zeros((L, len(net.products)))
        for k, (i, j) in enumerate(net.products):
            net.A[i:j, k] = 1.0
        net.span = np.array([j - i for (i, j) in net.products])
        net.fare = np.array(fare, dtype=float)
        net.lam = np.array(lam, dtype=float)
        net.T = net.lam.shape[1]
        net.D = net.lam.sum(axis=1)
        net.cum = np.cumsum(net.lam, axis=0)
        return net

    @property
    def long_mask(self):
        """Lange Fahrten: mehr als die halbe Linie (bei L = 4 ab drei Teilstrecken, bei L = 2 nur die ganze Linie)."""
        return 2 * self.span > self.L

    def forecast(self, err: float):
        """Prognose der Nachfrage: lange Fahrten mit dem Faktor 1 + err verzerrt, alles andere wahr. Rückgabe: lam_hat (wie lam)."""
        f = np.where(self.long_mask, 1.0 + err, 1.0)
        lam_hat = self.lam * f[:, None]
        s = lam_hat.sum(axis=0)
        if s.max() > C.MAX_ARRIVAL:
            lam_hat = lam_hat / s.max() * C.MAX_ARRIVAL
        return lam_hat


def build_net(p: Params) -> Net:
    return Net(p.L, p.cap, p.rho, p.alpha, p.early, C.GAMMA[p.kulisse])


def sample_sequence(net: Net, rng: SplitMix64) -> list:
    """Anfragefolge: je Periode der Produktindex oder -1 (keine Anfrage). Eine Ziehung je Periode."""
    out = []
    for t in range(net.T):
        u = rng.random()
        k = int(np.searchsorted(net.cum[:, t], u, side="right"))
        out.append(k if k < len(net.fare) else -1)
    return out
