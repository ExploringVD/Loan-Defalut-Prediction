"""Display helpers shared by the pages: probability text, decision banner, SHAP reasons chart."""
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Decision bands: colour AND icon AND words, so the meaning never depends on colour alone
DECISIONS = {
    "APPROVE": {"icon": "✅", "kind": "success", "color": "#0ca30c", "text": "low risk - the loan can be approved"},
    "REVIEW": {"icon": "⚠️", "kind": "warning", "color": "#fab219", "text": "medium risk - a credit officer should review it"},
    "REJECT": {"icon": "⛔", "kind": "error", "color": "#d03b3b", "text": "high risk - the loan should be rejected"},
}
INCREASES, DECREASES = "#d03b3b", "#2a78d6"     # red = increases risk, blue = decreases risk

HOME_LABELS = {"RENT": "Rents", "MORTGAGE": "Has a mortgage", "OWN": "Owns the home", "OTHER": "Other"}
INTENT_LABELS = {"EDUCATION": "Education", "MEDICAL": "Medical bills", "VENTURE": "Business venture",
                 "PERSONAL": "Personal", "DEBTCONSOLIDATION": "Debt consolidation",
                 "HOMEIMPROVEMENT": "Home improvement"}


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


def show_decision(decision: str, probability: float) -> None:
    d = DECISIONS[decision]
    message = f"**{decision}** - {d['text']}.  Probability of default: **{format_probability(probability)}**"
    getattr(st, d["kind"])(message, icon=d["icon"])


def reasons_figure(reasons: list[dict]) -> go.Figure:
    """Horizontal bars of the top SHAP reasons; the biggest reason is at the top."""
    ordered = list(reversed(reasons))
    fig = go.Figure(go.Bar(
        x=[r["contribution"] for r in ordered],
        y=[r["text"] for r in ordered],
        orientation="h",
        marker_color=[INCREASES if r["contribution"] > 0 else DECREASES for r in ordered],
        customdata=[[r["label"], "not given" if r["value"] is None else r["value"]] for r in ordered],
        hovertemplate="%{customdata[0]}: %{customdata[1]}<br>SHAP %{x:+.2f} (log-odds)<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#898781")
    fig.update_layout(
        title="Top 5 reasons (red = increases risk, blue = decreases risk)",
        xaxis_title="SHAP contribution (log-odds)",
        height=320, margin=dict(l=10, r=10, t=50, b=40), showlegend=False,
    )
    return fig


def show_result(application: dict) -> None:
    """Decision banner, reasons chart and reasons as sentences for one scored application."""
    p = application["prediction"]
    show_decision(p["decision"], p["probability"])
    left, right = st.columns([2, 3])
    with left:
        st.metric("Probability of default", format_probability(p["probability"]))
        st.markdown("**Why?**")
        for r in p["top_reasons"]:
            arrow = "🔺" if r["contribution"] > 0 else "🔻"
            st.markdown(f"{arrow} {r['text']}")
        st.caption(f"Application #{application['id']} for {application['applicant_name']} - model version "
                   f"{p['model_version']} - scored in {p['latency_ms']:.0f} ms")
    with right:
        st.plotly_chart(reasons_figure(p["top_reasons"]), width="stretch")


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
