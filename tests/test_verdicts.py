"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede im Text genannte Zahl steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt eine Analyse, nicht Seeds; so prüft der Test die Grenzen (> −1,0 %, > 2 Standardfehler) und jede Zahl im Text direkt."""

import tsb_evaluation as ev
from tsb_scenario import Params


def _rule(gap, se=0.5, exact=None, vs=0.0, vs_se=0.5, short=0.5, long=0.5, same=False):
    return ev.RuleStat(revenue=100.0 + gap, gap=ev.Stat(gap, se), vs_fcfs=ev.Stat(vs, vs_se), exact_gap=exact, acc_short=short, acc_long=long, same_as_fcfs=same)


def _analysis(params=Params(), fcfs=None, bid=None, resolve=None, hat=None, n=200):
    rules = {
        "fcfs": fcfs or _rule(-10.0, exact=-10.5, short=0.55, long=0.51, same=True),
        "bid_static": bid or _rule(-2.0, exact=-2.1, vs=8.0, vs_se=1.0),
        "bid_resolve": resolve or _rule(-0.5, exact=None, vs=9.5, vs_se=1.0),
        "dp_hat": hat or _rule(0.0, se=0.0, exact=0.0, short=0.86, long=0.02),
        "dp": _rule(0.0, se=0.0, exact=0.0, short=0.86, long=0.02),
    }
    return ev.Analysis(settings=ev.Settings(params=params), n=n, T=81, n_products=10, states=6561, v0=250.0, sim_vs_v0=ev.Stat(0.1, 0.2), rules=rules, examples={}, requested_long=3.0, requested_short=20.0, n_long_products=3)


# ---------------------------------------------------------------- Kernfrage 1
def test_fcfs_verdict_is_a_warning_with_numbers_and_acceptance_shares():
    kind, text = ev.verdict_fcfs(_analysis())
    assert kind == "warning"
    assert "90,0 %" in text and "Mittel über 200 Folgen" in text and "Standardfehler 0,5 Prozentpunkte" in text and "89,5 %" in text
    assert "der Länge 1 55 %" in text and "von den langen Fahrten 51 %" in text and "86 % und 2 %" in text


def test_fcfs_verdict_threshold_is_one_percent_of_the_exact_value():
    assert ev.verdict_fcfs(_analysis(fcfs=_rule(-1.0, exact=-1.01)))[0] == "warning"
    kind, text = ev.verdict_fcfs(_analysis(fcfs=_rule(-1.2, exact=-0.99)))
    assert kind == "info" and text.endswith("Der Verlust liegt unter einem Prozent.") and "konkurrieren" not in text      # keine Ursachenaussage: bei Last 3 und α = 1 gilt der Satz auch


def test_fcfs_verdict_uses_the_words_of_the_hotel():
    _kind, text = ev.verdict_fcfs(_analysis(params=Params(kulisse="hotel")))
    assert "von den langen Aufenthalten" in text


# ---------------------------------------------------------------- Kernfrage 2
def test_rules_verdict_names_the_three_levels_and_all_standard_errors():
    kind, text = ev.verdict_rules(_analysis())
    assert kind == "info"
    assert "Annahme nach Eingang 90,0 %" in text and "Bid-Price einmal gerechnet 98,0 %" in text and "je Periode neu 99,5 %" in text
    assert "(Standardfehler 0,5, 0,5, 0,5 Prozentpunkte)" in text
    assert "holt gegenüber der Annahme nach Eingang +8,0 Prozentpunkte (Standardfehler 1,0)" in text


def test_rules_verdict_warns_when_the_bid_price_is_clearly_worse_than_first_come():
    bid = _rule(-3.0, vs=-1.5, vs_se=0.5)                       # genau 3 Standardfehler darunter
    kind, text = ev.verdict_rules(_analysis(fcfs=_rule(-1.5, same=True), bid=bid))
    assert kind == "warning" and "schlechter als die Annahme nach Eingang (−1,5 Prozentpunkte, Standardfehler 0,5)" in text


def test_rules_verdict_two_standard_errors_are_not_enough_in_either_direction():
    for vs in (-1.0, 1.0):                                       # genau 2 Standardfehler
        kind, text = ev.verdict_rules(_analysis(bid=_rule(-2.0, vs=vs, vs_se=0.5)))
        assert kind == "info" and "nicht größer als zwei Standardfehler" in text


def test_rules_verdict_reports_identical_decisions():
    kind, text = ev.verdict_rules(_analysis(bid=_rule(-10.0, vs=0.0, vs_se=0.0, same=True)))
    assert kind == "info" and "in allen 200 Folgen genau dieselben Anfragen an wie die Annahme nach Eingang" in text


# ---------------------------------------------------------------- Kernfrage 3
def test_forecast_verdict_without_distortion_explains_the_regulator():
    kind, text = ev.verdict_forecast(_analysis())
    assert kind == "info" and "stimmt hier mit der wahren Nachfrage überein" in text and "um bis zu 50 %" in text


def test_forecast_verdict_with_distortion_names_every_rule_and_the_sign():
    hat = _rule(-0.3, se=0.1, exact=-0.25)
    kind, text = ev.verdict_forecast(_analysis(params=Params(err=0.5), hat=hat))
    assert kind == "info"
    assert "mit +50 % Nachfrage nach langen Fahrten" in text and "99,7 % (exakter Erwartungswert 99,8 %)" in text and "Bid-Price einmal gerechnet 98,0 %" in text and "(braucht keine Prognose) 90,0 %" in text
    assert ev.verdict_forecast(_analysis(params=Params(err=-0.25), hat=hat))[1].count("−25 %") == 1


def test_forecast_verdict_threshold_is_two_percent_of_the_exact_value():
    assert ev.verdict_forecast(_analysis(params=Params(err=0.5), hat=_rule(-2.0, exact=-2.0)))[0] == "info"
    assert ev.verdict_forecast(_analysis(params=Params(err=0.5), hat=_rule(-2.1, exact=-2.01)))[0] == "warning"


# ---------------------------------------------------------------- Messreihen-Zusammenfassungen
def _cell(cap, rho, alpha, early, fcfs, bid, vs, vs_se, same=False):
    return {"cap": cap, "rho": rho, "alpha": alpha, "early": early, "n": 1000, "fcfs": {"gap": fcfs, "exact_gap": fcfs}, "bid_static": {"gap": bid, "exact_gap": bid}, "bid_vs_fcfs": vs, "bid_vs_fcfs_se": vs_se, "bid_same_as_fcfs": same}


def test_raster_summary_counts_on_hand_made_cells():
    rows = [
        _cell(4, 0.8, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, same=True),
        _cell(4, 1.2, 0.6, 0.0, -0.5, -2.5, -2.0, 0.5),            # 4 Standardfehler schlechter
        _cell(8, 2.0, 0.3, 1.0, -20.0, -3.0, 17.0, 0.5),           # Überlast
        _cell(8, 2.0, 0.6, 0.0, -4.0, -2.0, 2.0, 1.0),             # genau 2 Standardfehler: weder besser noch schlechter
        _cell(8, 3.0, 1.0, 0.0, -2.0, -2.0, 0.0, 0.0, same=True),
    ]
    s = ev.raster_summary(rows)
    assert s["cells"] == 5 and s["fcfs_min"] == -20.0 and s["fcfs_max"] == -0.5 and s["bid_min"] == -3.0 and s["bid_max"] == -1.0
    assert s["low_load_fcfs_min"] == -1.0 and s["over_cells"] == 2 and s["over_fcfs_min"] == -20.0 and s["over_fcfs_max"] == -4.0
    assert s["over_bid_min"] == -3.0 and s["over_bid_max"] == -2.0
    assert s["a1_cells"] == 2 and s["a1_same"] == 2 and s["a1_fcfs_min"] == -2.0 and s["a1_fcfs_max"] == -1.0 and s["same_cells"] == 2
    assert len(s["worse"]) == 1 and s["worse"][0]["cap"] == 4 and s["better"] == 1
    assert s["worse_exact"] == 1          # nur die Zelle (-0,5 gegen -2,5) ist auch nach dem exakten Wert schlechter


def test_forecast_summary_takes_the_largest_loss():
    rows = [{"dp_hat": {"gap": -0.1, "exact_gap": -0.05}, "fcfs": {"gap": -13.0}}, {"dp_hat": {"gap": -0.5, "exact_gap": -0.4}, "fcfs": {"gap": -1.2}}]
    s = ev.forecast_summary(rows)
    assert s["rows"] == 2 and s["dp_hat_worst"] == 0.4 and s["dp_hat_worst_sim"] == 0.5 and s["fcfs_min"] == -13.0 and s["fcfs_max"] == -1.2
