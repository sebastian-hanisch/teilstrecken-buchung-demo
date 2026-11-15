"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsb_scenario import Net  # noqa: E402


@pytest.fixture
def mini_certain():
    """Zwei Teilstrecken mit je einem Platz, Produkte (0,1) für 10, (0,2) für 15, (1,2) für 10; zwei Perioden.
    Periode 1: Produkt (0,1) mit Wahrscheinlichkeit 0,5; Periode 2: Produkt (0,2) sicher. Handrechnung:
    nach Eingang 0,5 · 10 + 0,5 · 15 = 12,5; exakte Rechnung 15 (die kurze Fahrt ablehnen, die lange kommt sicher)."""
    lam = [[0.5, 0.0],    # Produkt (0, 1)
           [0.0, 1.0],    # Produkt (0, 2)
           [0.0, 0.0]]    # Produkt (1, 2)
    return Net.from_arrays(2, 1, [10.0, 15.0, 10.0], lam)


@pytest.fixture
def mini_flat():
    """Dieselben Preise, zwei Perioden mit gleichen Wahrscheinlichkeiten 0,2 / 0,5 / 0,2 je Periode. Handrechnung: V_2(1,1) = 11,5; V_1(1,1) = 13,45."""
    lam = [[0.2, 0.2], [0.5, 0.5], [0.2, 0.2]]
    return Net.from_arrays(2, 1, [10.0, 15.0, 10.0], lam)
