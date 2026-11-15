"""Kern gegen Handrechnung (DEMO-PLAYBOOK §1b): exakte Rechnung, Annahmeregeln, Zulässigkeit und Folgenerzeuger an Mini-Instanzen, deren Werte von Hand gerechnet sind."""

import numpy as np
import pytest

import tsb_algorithm as A
import tsb_constants as C
from tsb_rng import SplitMix64
from tsb_scenario import Net, Params, build_net, cap_max, n_states, sample_sequence


# ---------------------------------------------------------------- Handrechnung (zwei Teilstrecken, ein Platz)
def test_dp_value_matches_the_hand_calculation(mini_flat):
    V = A.dp_values(mini_flat, mini_flat.lam)
    # V[2] = 0 (nach der letzten Periode); V[1] (vor der zweiten Periode) bei (1,1): 0,2 · 10 + 0,5 · 15 + 0,2 · 10 = 11,5; an den Rändern passt nur die kurze Fahrt der freien Teilstrecke
    assert V[1][1, 1] == pytest.approx(11.5)
    assert V[1][1, 0] == pytest.approx(2.0) and V[1][0, 1] == pytest.approx(2.0) and V[1][0, 0] == pytest.approx(0.0)
    # V[0](1,1) = 11,5 + 0,2 · max(0, 10 + 2 - 11,5) + 0,5 · max(0, 15 + 0 - 11,5) + 0,2 · max(0, 10 + 2 - 11,5) = 11,5 + 0,1 + 1,75 + 0,1
    assert V[0][1, 1] == pytest.approx(13.45)


def test_exact_rule_rejects_the_short_trip_when_a_certain_long_trip_follows(mini_certain):
    ctx = A.Context(mini_certain, mini_certain.lam)
    assert ctx.V[0][1, 1] == pytest.approx(15.0)
    assert A.accepts(ctx, "dp", 0, [1, 1], 0) is False           # 10 + V_2(0,1) = 10 < V_2(1,1) = 15
    assert A.accepts(ctx, "dp", 1, [1, 1], 1) is True            # die lange Fahrt kommt, Platz ist da
    assert A.accepts(ctx, "fcfs", 0, [1, 1], 0) is True


def test_policies_on_the_two_period_example(mini_certain):
    ctx = A.Context(mini_certain, mini_certain.lam)
    seq = [0, 1]                                                  # erst die kurze, dann die lange Fahrt
    assert A.run_policy(mini_certain, seq, "fcfs", ctx)[0] == pytest.approx(10.0)    # die kurze belegt Teilstrecke 1, die lange passt nicht mehr
    assert A.run_policy(mini_certain, seq, "dp", ctx)[0] == pytest.approx(15.0)
    assert A.run_policy(mini_certain, [-1, 1], "fcfs", ctx)[0] == pytest.approx(15.0)


def test_exact_expected_value_of_the_rules_by_hand(mini_certain):
    ctx = A.Context(mini_certain, mini_certain.lam)
    assert A.value_of_policy(mini_certain, "fcfs") == pytest.approx(12.5)           # 0,5 · 10 + 0,5 · 15
    assert A.value_of_policy(mini_certain, "dp", V_hat=ctx.V) == pytest.approx(15.0)


def test_bid_prices_on_a_unique_dual_instance():
    # Preise 10 / 15 / 10, je Teilstrecke ein Platz, Nachfrage reichlich: die beiden kurzen Fahrten (20) schlagen die lange (15); Schattenpreise (10, 10), eindeutig
    net = Net.from_arrays(2, 1, [10.0, 15.0, 10.0], [[5.0], [5.0], [5.0]])
    pi = A.bid_prices(net, [1, 1], net.D)
    assert pi == pytest.approx([10.0, 10.0])
    ctx = A.Context(net, net.lam)
    assert A.accepts(ctx, "bid_static", 0, [1, 1], 1) is False    # 15 < 10 + 10: die lange Fahrt wird abgelehnt
    assert A.accepts(ctx, "bid_static", 0, [1, 1], 0) is True     # 10 >= 10


# ---------------------------------------------------------------- Zulässigkeit
@pytest.mark.parametrize("params", [Params(), Params(L=3, cap=3, rho=3.0, alpha=0.3), Params(L=2, cap=2, rho=1.0, alpha=1.0, early=0.0), Params(kulisse="hotel", L=5, cap=3, rho=2.5)])
def test_no_rule_ever_exceeds_the_capacity(params):
    net = build_net(params)
    ctx = A.Context(net, net.lam)
    for k in range(6):
        seq = sample_sequence(net, SplitMix64(1000 + k))
        for rule in C.RULES:
            rev, acc, req, events = A.run_policy(net, seq, rule, ctx)
            occ = A.occupancy_by_span(net, events)
            assert occ.sum(axis=1).max() <= net.cap
            assert (acc <= req).all()
            assert rev == pytest.approx(sum(net.fare[p] for (_t, p, take) in events if take))


def test_the_exact_rule_is_never_worse_than_the_others_in_expectation():
    net = build_net(Params(L=3, cap=3, rho=2.0, alpha=0.3))
    ctx = A.Context(net, net.lam)
    best = A.value_of_policy(net, "dp", V_hat=ctx.V)
    assert best == pytest.approx(float(ctx.V[0][(net.cap,) * net.L]))
    assert A.value_of_policy(net, "fcfs") <= best + 1e-9
    assert A.value_of_policy(net, "bid_static", pi=ctx.pi_static) <= best + 1e-9


# ---------------------------------------------------------------- Szenario
def test_default_params_equal_the_first_preset_and_the_constants():
    p = Params()
    assert (p.kulisse, p.L, p.cap, p.rho, p.alpha, p.early, p.err) == (C.DEFAULT_KULISSE, C.DEFAULT_L, C.DEFAULT_CAP, C.DEFAULT_RHO, C.DEFAULT_ALPHA, C.DEFAULT_EARLY, C.DEFAULT_ERR)


def test_net_structure_for_four_segments():
    net = build_net(Params())
    assert len(net.products) == 10 and net.T == 81
    assert net.products[0] == (0, 1) and net.products[-1] == (3, 4)
    assert net.fare[0] == pytest.approx(10.0)                                  # Länge 1
    assert net.fare[net.products.index((0, 4))] == pytest.approx(10.0 * 4 ** 0.3)
    assert net.long_mask.sum() == 3                                            # (0,3), (1,4), (0,4): mehr als die halbe Linie


def test_arrival_probabilities_are_valid_and_the_load_is_rho_times_capacity():
    for params in (Params(), Params(rho=3.0, early=0.5), Params(L=6, cap=6), Params(kulisse="hotel", L=5)):
        net = build_net(params)
        assert (net.lam >= 0).all() and net.lam.sum(axis=0).max() <= C.MAX_ARRIVAL + 1e-12
        per_segment = net.A @ net.D                                            # erwartete Nachfrage je Teilstrecke
        assert per_segment.mean() == pytest.approx(params.rho * params.cap, rel=1e-9)


def test_early_booking_moves_long_trips_to_the_beginning():
    net0, net1 = build_net(Params(early=0.0)), build_net(Params(early=1.0))
    long_p = int(np.argmax(net1.span))
    half = net1.T // 2
    assert net0.lam[long_p, :half].sum() == pytest.approx(net0.lam[long_p, half:].sum(), rel=0.05)
    assert net1.lam[long_p, :half].sum() > 1.5 * net1.lam[long_p, half:].sum()
    assert net0.D == pytest.approx(net1.D, rel=1e-9)                            # nur der zeitliche Verlauf ändert sich, nicht die Gesamtnachfrage


def test_forecast_distorts_only_long_trips():
    net = build_net(Params())
    hat = net.forecast(0.5)
    ratio = hat.sum(axis=1) / net.lam.sum(axis=1)
    assert ratio[net.long_mask] == pytest.approx(1.5) and ratio[~net.long_mask] == pytest.approx(1.0)
    assert net.forecast(0.0) == pytest.approx(net.lam)


def test_sequences_are_reproducible_and_follow_the_probabilities():
    net = build_net(Params(L=2, cap=2, rho=1.0))
    assert sample_sequence(net, SplitMix64(7)) == sample_sequence(net, SplitMix64(7))
    counts = np.zeros(len(net.products))
    none = 0
    for k in range(3000):
        for p in sample_sequence(net, SplitMix64(500 + k)):
            if p < 0:
                none += 1
            else:
                counts[p] += 1
    n = 3000
    assert counts / n == pytest.approx(net.lam.sum(axis=1), abs=0.04 * net.lam.sum(axis=1).max() + 0.05)
    assert none / n == pytest.approx((1 - net.lam.sum(axis=0)).sum(), rel=0.03)


def test_capacity_bound_keeps_the_exact_rule_live():
    for L in range(C.L_MIN, C.L_MAX + 1):
        c = cap_max(L)
        assert C.CAP_MIN <= c <= C.CAP_MAX and n_states(L, c) <= C.STATES_LIVE_MAX
        assert c == C.CAP_MAX or n_states(L, c + 1) > C.STATES_LIVE_MAX
    assert (cap_max(4), cap_max(5), cap_max(6)) == (16, 9, 6)
