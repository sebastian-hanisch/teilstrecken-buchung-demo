"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass, fields

import streamlit as st

import tsb_constants as C
from tsb_scenario import Params, cap_max

DEFAULTS = Params()


@dataclass(frozen=True)
class SettingSpec:
    param: str       # Feld von Params ("seed" und "rule" gehören nicht zum Szenario)
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "kulisse_select": SettingSpec("kulisse", "kulisse", str, DEFAULTS.kulisse, choices=C.KULISSEN),
    "L_slider": SettingSpec("L", "L", int, DEFAULTS.L, C.L_MIN, C.L_MAX),
    "cap_slider": SettingSpec("cap", "cap", int, DEFAULTS.cap, C.CAP_MIN, C.CAP_MAX),
    "rho_select": SettingSpec("rho", "rho", float, DEFAULTS.rho, choices=C.RHO_STEPS),
    "alpha_select": SettingSpec("alpha", "alpha", float, DEFAULTS.alpha, choices=C.ALPHA_STEPS),
    "early_select": SettingSpec("early", "early", float, DEFAULTS.early, choices=C.EARLY_STEPS),
    "err_select": SettingSpec("err", "err", float, DEFAULTS.err, choices=C.ERR_STEPS),
    "rule_select": SettingSpec("rule", "rule", str, C.DEFAULT_RULE, choices=C.RULES),
    "seed_input": SettingSpec("seed", "seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PARAM_TO_STATE = {spec.param: key for key, spec in SETTING_SPECS.items()}
VIEW_ONLY = ("rule_select", "seed_input")            # ändern die Ansicht bzw. die Folgen, nicht das Szenario: die Preset-Markierung bleibt
assert {f.name for f in fields(Params)} | {"seed", "rule"} == set(PARAM_TO_STATE), "jedes Feld von Params braucht einen Regler"


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (Auswahlfelder: nur gültige Werte)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None:
                if value not in spec.choices:
                    continue
            else:
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def clamp_capacity():
    """Die Kapazität darf höchstens so groß sein, dass die exakte Rechnung bei L Teilstrecken live bleibt; nach Permalink, Preset oder neuem L anwenden, bevor der Regler gezeichnet wird."""
    L = st.session_state.get("L_slider", DEFAULTS.L)
    cap = st.session_state.get("cap_slider", DEFAULTS.cap)
    st.session_state["cap_slider"] = max(C.CAP_MIN, min(cap, cap_max(L)))


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt jeden Szenario-Regler: genannte Felder aus dem Preset, alle übrigen auf den Standard (Ansicht und Zufalls-Seed bleiben)."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key in VIEW_ONLY:
            continue
        st.session_state[state_key] = C.PRESETS[name].get(spec.param, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen (Ansicht und Zufalls-Seed ausgenommen), sonst None."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(key, spec.default) == preset.get(spec.param, spec.default)
               for key, spec in SETTING_SPECS.items() if key not in VIEW_ONLY):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
                  type="primary" if aktiv == name else "secondary")


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
