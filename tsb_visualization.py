"""Plotly-Figuren: Belegung der Teilstrecken nach Fahrtlänge und Vergleich der Regeln (alle Achsen fest)."""

import plotly.graph_objects as go

import tsb_constants as C


def _layout(fig, height, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10), title=title, legend=dict(orientation="h", y=-0.25))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_occupancy(example, params, title):
    """Belegte Plätze je Teilstrecke am Ende der Folge, gestapelt nach Länge der Fahrt; rote Linie = Kapazität."""
    w = C.WORDS[params.kulisse]
    L, cap = params.L, params.cap
    fig = go.Figure()
    xs = [f"{w['segment']} {i + 1}" for i in range(L)]
    for s in range(L):
        ys = [example.occupancy[l][s] for l in range(L)]
        fig.add_bar(x=xs, y=ys, name=f"Länge {s + 1}", marker_color=C.LENGTH_COLORS[s], hovertemplate="%{x}: %{y} bei Länge " + str(s + 1) + "<extra></extra>")
    fig.add_hline(y=cap, line_dash="dash", line_color=C.COLOR_LIMIT, annotation_text=f"Kapazität {cap}", annotation_position="top left")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(range=[0, cap * 1.15], title=f"belegte {w['units']}")
    return _layout(fig, 330, title)


def build_rules(analysis, params):
    """Ertrag der Regeln in Prozent der exakten Rechnung (mit wahrer Nachfrage = 100 %), Fehlerbalken = Standardfehler."""
    keys = ["fcfs", "bid_static", "bid_resolve"] + (["dp_hat"] if params.err != 0.0 else [])
    labels = {**C.RULE_LABELS, "dp_hat": "Exakte Rechnung mit der falschen Prognose"}
    vals = [100.0 + analysis.rules[k].gap.mean for k in keys]
    errs = [analysis.rules[k].gap.se for k in keys]
    names = [f"{labels[k]}: {v:.1f} %".replace(".", ",") for k, v in zip(keys, vals)]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[C.COLOR_BARS[k] for k in keys], error_x=dict(type="data", array=errs, visible=True),
                           hovertemplate="%{y}<extra></extra>"))
    fig.add_vline(x=100, line_dash="dash", line_color=C.COLOR_BARS["dp"], annotation_text="exakte Rechnung = 100 %", annotation_position="top")
    lo = max(0.0, min(vals) - 8.0)
    fig.update_xaxes(range=[lo, 103], title="Ertrag in % der exakten Rechnung")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 260)
