"""Display helpers shared by the pages: theme (colours, CSS, Plotly), page header, probability text, decision badge,
probability gauge and SHAP reasons. All styling lives in style.css + .streamlit/config.toml - not in the views."""
import html
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api import ApiClient, ApiError

# Colour theme (the same values are in .streamlit/config.toml)
ICE, AUBERGINE, MARIGOLD, WHITE = "#DDE3E8", "#3A0F34", "#EAAA00", "#FFFFFF"
GREEN, CRIMSON = "#2E8B57", "#C0392B"
FONT = "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
STYLE_FILE = Path(__file__).parent / "style.css"
LOGO_FILE = Path(__file__).parent / "assets" / "logo.svg"      # sidebar logo
ICON_FILE = Path(__file__).parent / "assets" / "icon.svg"      # logo when the sidebar is collapsed

# Decision bands: colour AND icon AND words, so the meaning never depends on colour alone.
# "ink" = text colour on the band colour (aubergine on marigold, white on green / crimson).
DECISIONS = {
    "APPROVE": {"icon": "✅", "color": GREEN, "ink": WHITE, "pill": "green",
                "text": "low risk - the loan can be approved"},
    "REVIEW": {"icon": "⚠️", "color": MARIGOLD, "ink": AUBERGINE, "pill": "yellow",
               "text": "medium risk - a credit officer should review it"},
    "REJECT": {"icon": "⛔", "color": CRIMSON, "ink": WHITE, "pill": "red",
               "text": "high risk - the loan should be rejected"},
}
INCREASES, DECREASES = CRIMSON, GREEN     # SHAP reasons: crimson = increases risk, green = decreases risk

HOME_LABELS = {"RENT": "Rents", "MORTGAGE": "Has a mortgage", "OWN": "Owns the home", "OTHER": "Other"}
INTENT_LABELS = {"EDUCATION": "Education", "MEDICAL": "Medical bills", "VENTURE": "Business venture",
                 "PERSONAL": "Personal", "DEBTCONSOLIDATION": "Debt consolidation",
                 "HOMEIMPROVEMENT": "Home improvement"}


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _css() -> str:
    return STYLE_FILE.read_text()


def apply_theme() -> None:
    """Load style.css (read from disk once). Called at the top of every run by app.py."""
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str) -> None:
    """Aubergine banner with a thin marigold line: small label, title, one-line subtitle."""
    with st.container(key="page_header"):
        st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
        st.title(title, anchor=False)
        st.caption(subtitle)


def style_figure(fig: go.Figure, title: str | None = None, height: int = 320) -> go.Figure:
    """One look for every Plotly chart: project font and colours, transparent background, light gridlines."""
    fig.update_layout(
        template="none", height=height,
        font=dict(family=FONT, color=AUBERGINE, size=13),
        title=dict(text=title or "", x=0, xanchor="left", font=dict(size=15, color=AUBERGINE)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50 if title else 10, b=30),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=AUBERGINE, font=dict(family=FONT, color=AUBERGINE)),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=12)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(58,15,52,0.25)", automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(58,15,52,0.08)", zeroline=False, automargin=True)
    return fig


def show_chart(fig: go.Figure, where=st) -> None:
    """Draw a styled chart (theme=None: our style_figure look instead of Streamlit's default theme)."""
    where.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def format_probability(p: float | None) -> str:
    """The calibrated model returns exactly 0 or 1 for some applicants - never show those as 0% / 100%."""
    if p is None:
        return "-"
    if p < 0.01:
        return "< 1%"
    if p > 0.99:
        return "> 99%"
    return f"{p:.1%}"


def format_time(iso: str) -> str:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")


def decision_label(decision: str) -> str:
    return f"{DECISIONS[decision]['icon']} {decision}"


def safe_html(text) -> str:
    """Escape text for our HTML snippets; '$' as an entity so Streamlit never reads '$...$' as maths."""
    return html.escape(str(text)).replace("$", "&#36;")


# ---------------------------------------------------------------------------
# Result: decision badge, probability gauge, reasons
# ---------------------------------------------------------------------------

def decision_badge(decision: str) -> str:
    d = DECISIONS[decision]
    return (f'<div class="decision-badge" style="--band:{d["color"]}; --ink:{d["ink"]}">'
            f'<span>{d["icon"]}</span> {decision}</div>'
            f'<div class="decision-text">{safe_html(d["text"]).capitalize()}</div>')


def decision_cutoffs() -> dict | None:
    """APPROVE / REVIEW / REJECT cut-offs from the API (model_meta.json), fetched once per session."""
    if "decision_cutoffs" not in st.session_state:
        try:
            st.session_state["decision_cutoffs"] = ApiClient(st.session_state.get("token")).model_info()["decision_cutoffs"]
        except ApiError:
            return None
    return st.session_state["decision_cutoffs"]


def probability_gauge(probability: float, cutoffs: dict | None) -> go.Figure:
    """Gauge 0-100% with the three bands coloured (green / marigold / crimson) and the applicant's probability."""
    steps = []
    if cutoffs:
        review, reject = cutoffs["review"] * 100, cutoffs["reject"] * 100
        steps = [dict(range=[0, review], color=GREEN), dict(range=[review, reject], color=MARIGOLD),
                 dict(range=[reject, 100], color=CRIMSON)]
    fig = go.Figure(go.Indicator(
        mode="gauge", value=probability * 100,
        gauge=dict(axis=dict(range=[0, 100], ticksuffix="%", tickcolor=AUBERGINE, tickwidth=1),
                   bar=dict(color=AUBERGINE, thickness=0.28), bgcolor=WHITE, borderwidth=0, steps=steps),
        domain=dict(x=[0, 1], y=[0.12, 1]),
    ))
    # The number is an annotation, so exact 0 / 1 read "< 1%" / "> 99%" like everywhere else
    fig.add_annotation(text=f"<b>{format_probability(probability)}</b>", x=0.5, y=0.12, showarrow=False,
                       font=dict(size=30, color=AUBERGINE, family=FONT))
    fig.add_annotation(text="probability of default", x=0.5, y=-0.04, showarrow=False,
                       font=dict(size=12, color=AUBERGINE, family=FONT))
    style_figure(fig, height=230)
    fig.update_layout(margin=dict(l=40, r=40, t=15, b=10))     # room for the 0% / 100% labels
    return fig


def reason_arrow(contribution: float) -> str:
    """Up arrow = increases risk, down arrow = decreases risk."""
    return "▲" if contribution > 0 else "▼"


def reason_chip(reason: dict) -> str:
    """One SHAP reason as a small card. Same colours as the bar chart (INCREASES / DECREASES)."""
    color = INCREASES if reason["contribution"] > 0 else DECREASES
    return (f'<div class="reason-chip" style="--chip:{color}"><span class="arrow">'
            f'{reason_arrow(reason["contribution"])}</span><span>{safe_html(reason["text"])}</span></div>')


def reasons_figure(reasons: list[dict]) -> go.Figure:
    """Horizontal bars of the top SHAP reasons; the biggest reason is at the top."""
    ordered = list(reversed(reasons))
    fig = go.Figure(go.Bar(
        x=[r["contribution"] for r in ordered],
        y=[r["text"] for r in ordered],
        orientation="h",
        marker=dict(color=[INCREASES if r["contribution"] > 0 else DECREASES for r in ordered], cornerradius=6),
        customdata=[[r["label"], "not given" if r["value"] is None else r["value"]] for r in ordered],
        hovertemplate="%{customdata[0]}: %{customdata[1]}<br>SHAP %{x:+.2f} (log-odds)<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color=AUBERGINE)
    style_figure(fig, "Top 5 reasons", height=320)     # the colour key is a caption under it (wraps on small screens)
    fig.update_xaxes(title_text="SHAP contribution", showgrid=True, gridcolor="rgba(58,15,52,0.08)")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(showlegend=False)
    return fig


def show_result(application: dict) -> None:
    """Decision badge + gauge, reasons as cards and as a chart, for one scored application."""
    p = application["prediction"]
    with st.container(key="result_card"):
        badge, gauge = st.columns([1, 1.3], vertical_alignment="center")
        with badge:
            st.markdown(decision_badge(p["decision"]), unsafe_allow_html=True)
            st.metric("Probability of default", format_probability(p["probability"]))
            st.caption(f"Application #{application['id']} for {application['applicant_name']} - model version "
                       f"{p['model_version']} - scored in {p['latency_ms']:.0f} ms")
        show_chart(probability_gauge(p["probability"], decision_cutoffs()), gauge)

        reasons, chart = st.columns([2, 3])
        chips = "".join(reason_chip(r) for r in p["top_reasons"])
        reasons.markdown(f'<div class="card-title">Why?</div><div class="reasons">{chips}</div>',
                         unsafe_allow_html=True)
        show_chart(reasons_figure(p["top_reasons"]), chart)
        chart.caption("Crimson ▲ = increases risk, green ▼ = decreases risk. Bar length = SHAP contribution "
                      "(log-odds of the model).")


def decision_donut(mix: dict, title: str, unit: str = "applications") -> go.Figure:
    """Donut of APPROVE / REVIEW / REJECT counts in the decision colours, total in the middle."""
    total = sum(mix.values())
    mix = {d: n for d, n in mix.items() if n}          # no "0%" slice labels
    fig = go.Figure(go.Pie(
        labels=[decision_label(d) for d in mix], values=list(mix.values()), hole=0.62, sort=False,
        marker=dict(colors=[DECISIONS[d]["color"] for d in mix], line=dict(color=WHITE, width=3)),
        textinfo="percent", insidetextorientation="horizontal",
        textfont=dict(color=[DECISIONS[d]["ink"] for d in mix]),
        hovertemplate=f"%{{label}}: %{{value:,}} {unit}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{total:,}</b><br>{unit}", showarrow=False, font=dict(size=15))
    return style_figure(fig, title, height=330)


def decision_pill(decision: str | None) -> list[str]:
    """Table cell value for decision_pill_column(): a one-item list shown as a coloured pill."""
    return [decision_label(decision)] if decision else []


def decision_pill_column(label: str = "Decision"):
    """st.dataframe column that shows the decision as a pill in the decision colour (theme green / yellow / red)."""
    return st.column_config.MultiselectColumn(label, options=[decision_label(d) for d in DECISIONS],
                                              color=[v["pill"] for v in DECISIONS.values()])


def applicant_table(application: dict) -> pd.DataFrame:
    """The 11 inputs of an application as a readable two-column table."""
    a = application
    rows = {
        "Age": a["person_age"], "Yearly income": f"${a['person_income']:,.0f}",
        "Home ownership": HOME_LABELS.get(a["person_home_ownership"], a["person_home_ownership"]),
        "Years employed": "not given" if a["person_emp_length"] is None else f"{a['person_emp_length']:g}",
        "Loan purpose": INTENT_LABELS.get(a["loan_intent"], a["loan_intent"]),
        "Loan amount": f"${a['loan_amnt']:,.0f}", "Loan grade": a["loan_grade"],
        "Interest rate": "not given" if a["loan_int_rate"] is None else f"{a['loan_int_rate']:g}%",
        "Loan as % of income": f"{a['loan_percent_income']:.0%}",
        "Past default on file": "Yes" if a["cb_person_default_on_file"] == "Y" else "No",
        "Credit history": f"{a['cb_person_cred_hist_length']} years",
    }
    return pd.DataFrame({"Field": list(rows), "Value": [str(v) for v in rows.values()]})
