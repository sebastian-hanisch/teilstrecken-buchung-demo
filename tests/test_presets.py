"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und was jedes Preset tatsächlich ändert."""

import dataclasses

import tsb_constants as C
import tsb_evaluation as ev
import tsb_presets as P
from tsb_scenario import Params, cap_max


def test_every_preset_has_help_and_only_known_fields():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    fields = {f.name for f in dataclasses.fields(Params)}
    for name, preset in C.PRESETS.items():
        assert set(preset) <= fields, name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regulator_ranges():
    for name, preset in C.PRESETS.items():
        p = Params(**preset)
        assert p.kulisse in C.KULISSEN and C.L_MIN <= p.L <= C.L_MAX, name
        assert C.CAP_MIN <= p.cap <= cap_max(p.L), name                 # die Kapazität bleibt unter der Grenze der Live-Rechnung
        assert p.rho in C.RHO_STEPS and p.alpha in C.ALPHA_STEPS and p.early in C.EARLY_STEPS and p.err in C.ERR_STEPS, name


def test_default_preset_is_the_first_and_equals_the_default_params():
    first = next(iter(C.PRESETS))
    assert Params(**C.PRESETS[first]) == Params()
    assert ev.Settings(params=Params(**C.PRESETS[first])) == ev.Settings()


def test_all_presets_are_pairwise_different():
    ps = [Params(**p) for p in C.PRESETS.values()]
    assert len({dataclasses.astuple(p) for p in ps}) == len(ps)


def test_setting_specs_cover_every_field_and_the_view_regulators():
    assert set(P.PARAM_TO_STATE) == {f.name for f in dataclasses.fields(Params)} | {"seed", "rule"}
    assert set(P.VIEW_ONLY) == {"rule_select", "seed_input"}


def test_regulator_choices_contain_the_defaults():
    assert P.DEFAULTS.rho in C.RHO_STEPS and P.DEFAULTS.alpha in C.ALPHA_STEPS and P.DEFAULTS.early in C.EARLY_STEPS and P.DEFAULTS.err in C.ERR_STEPS
    assert 0.0 in C.ERR_STEPS and 1.0 in C.ALPHA_STEPS and 0.0 in C.EARLY_STEPS


def test_the_kulisse_changes_the_numbers_not_only_the_words():
    a = ev.analyse(ev.Settings(params=Params(kulisse="zug")), n_seq=20)
    b = ev.analyse(ev.Settings(params=Params(kulisse="hotel")), n_seq=20)
    assert a.v0 != b.v0 and a.rules["fcfs"].exact_gap != b.rules["fcfs"].exact_gap
