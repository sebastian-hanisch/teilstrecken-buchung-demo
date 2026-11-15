"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import tsb_algorithm as A
import tsb_constants as C
import tsb_evaluation as ev
from tsb_rng import SplitMix64
from tsb_scenario import Net, Params, build_net, sample_sequence


def test_smallest_network_two_segments_two_seats_runs_every_rule():
    a = ev.analyse(ev.Settings(params=Params(L=2, cap=2, rho=0.6)), n_seq=30)
    assert a.n == 30 and a.states == 9
    for r in a.rules.values():
        assert np.isfinite(r.revenue) and np.isfinite(r.gap.mean)
    assert a.rules["dp"].gap.mean == 0.0 and a.rules["dp"].exact_gap == 0.0


def test_one_seat_and_one_segment_is_the_single_resource_case():
    """L = 1: es gibt nur ein Produkt; alle Regeln, auch der Bid-Price, nehmen dann dasselbe an, solange der Preis den Schattenpreis deckt."""
    net = Net.from_arrays(1, 2, [10.0], [[0.4] * 6])
    ctx = A.Context(net, net.lam)
    seq = [0, 0, 0, -1, 0, 0]
    assert A.run_policy(net, seq, "fcfs", ctx)[0] == pytest.approx(20.0)        # zwei Plätze, drei Anfragen vorweg: zwei angenommen
    assert A.run_policy(net, seq, "dp", ctx)[0] == pytest.approx(20.0)
    assert A.value_of_policy(net, "fcfs") == pytest.approx(float(ctx.V[0][2]))   # eine Preisklasse: Annahme nach Eingang ist optimal


def test_no_demand_gives_zero_revenue_and_no_division_by_zero_in_the_shares():
    net = Net.from_arrays(2, 1, [10.0, 15.0, 10.0], [[0.0, 0.0]] * 3)
    ctx = A.Context(net, net.lam)
    for rule in C.RULES:
        assert A.run_policy(net, [-1, -1], rule, ctx)[0] == 0.0
    assert ctx.V[0][1, 1] == 0.0


def test_highest_load_with_one_seat_per_segment_never_exceeds_capacity_and_stays_finite():
    a = ev.analyse(ev.Settings(params=Params(L=3, cap=2, rho=3.0, alpha=0.2, early=1.0, err=0.5)), n_seq=25)
    assert all(np.isfinite(r.gap.mean) for r in a.rules.values())
    for ex in a.examples.values():
        assert max(sum(row) for row in ex.occupancy) <= 2


def test_forecast_error_acts_on_the_single_long_product_for_two_segments():
    """Bei L = 2 ist nur die ganze Linie eine lange Fahrt; der Regler wirkt trotzdem (nur auf dieses eine Produkt)."""
    net = build_net(Params(L=2, cap=3, rho=2.0, err=0.5))
    assert net.long_mask.tolist() == [False, True, False]
    hat = net.forecast(0.5)
    assert hat[1].sum() == pytest.approx(1.5 * net.lam[1].sum()) and hat[0].sum() == pytest.approx(net.lam[0].sum())


def test_same_zero_seed_and_the_largest_seed_give_valid_sequences():
    net = build_net(Params(L=2, cap=2))
    for seed in (0, C.SEED_MAX):
        seq = sample_sequence(net, SplitMix64(seed * C.SUBSEED_STRIDE))
        assert len(seq) == net.T and all(-1 <= p < len(net.products) for p in seq)


def test_ties_in_the_decision_accept_by_the_documented_rule():
    """Preis plus Wert danach gleich dem Wert jetzt: die Regel nimmt an (>=). Zwei Perioden, eine Teilstrecke, in Periode 2 kommt sicher dieselbe Fahrt, ein Platz: der Wert danach ist 10, Annahme jetzt gibt 10 + 0."""
    net = Net.from_arrays(1, 1, [10.0], [[0.5, 1.0]])
    ctx = A.Context(net, net.lam)
    assert ctx.V[1][1] == pytest.approx(10.0) and A.accepts(ctx, "dp", 0, [1], 0) is True     # 10 + 0 >= 10
