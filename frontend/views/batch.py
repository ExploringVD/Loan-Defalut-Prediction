"""Batch Scoring: upload a CSV of applicants, score it with POST /applications/batch, download the results.
Nothing is stored in the database (the API writes one audit-log entry per file)."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from api import ApiClient, ApiError

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "assets" / "sample_applicants.csv"   # 10 real test-split loans
REQUIRED = ["person_age", "person_income", "person_home_ownership", "loan_intent", "loan_grade", "loan_amnt",
            "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"]
OPTIONAL = ["person_emp_length", "loan_int_rate", "loan_id", "applicant_name", "loan_status"]


def frame(rows: list[dict]) -> pd.DataFrame:
    """API rows -> table without columns that are empty everywhere (e.g. applicant_name when the file has none)."""
    return pd.DataFrame(rows).dropna(axis=1, how="all")


def run(filename: str, content: bytes) -> None:
    """Send the file to the API and keep the result for the redraw."""
    try:
        with st.spinner(f"Scoring {filename}..."):
            result = ApiClient(st.session_state["token"]).score_batch(filename, content)
    except ApiError as e:
        if e.status == 401:
            raise                       # app.guarded() sends the user back to the login page
        st.session_state["batch_result"] = None
        st.session_state["batch_error"] = e.message
        return
    st.session_state.update(batch_result={"file": filename, **result}, batch_error=None)
    st.toast(f"{filename}: {result['summary']['rows_scored']:,} rows scored", icon=":material/task_alt:")


def kpi_tiles(summary: dict) -> None:
    ev = summary["evaluation"]
    c = st.columns(4)
    with c[0], st.container(key="kpi_batch_scored"):
        st.metric("Rows scored", f"{summary['rows_scored']:,}", delta=f"of {summary['rows_total']:,} in the file",
                  delta_color="off", delta_arrow="off")
    with c[1], st.container(key="kpi_reject"):
        st.metric("Rows rejected", f"{summary['rows_rejected']:,}", delta="broke a validation rule",
                  delta_color="off", delta_arrow="off")
    with c[2], st.container(key="kpi_latency"):
        speed = summary["rows_total"] / summary["seconds"] if summary["seconds"] else 0
        st.metric("Time taken", f"{summary['seconds']:.1f} s", delta=f"{speed:,.0f} rows per second",
                  delta_color="off", delta_arrow="off")
    with c[3], st.container(key="kpi_auc"):
        if ev and ev["roc_auc"] is not None:
            st.metric("ROC-AUC", f"{ev['roc_auc']:.3f}", delta=f"on {ev['rows_with_loan_status']:,} rows with loan_status",
                      delta_color="off", delta_arrow="off")
        else:
            st.metric("Model version", summary["model_version"], delta="no loan_status column",
                      delta_color="off", delta_arrow="off")


def default_rate_chart(evaluation: dict) -> go.Figure:
    """Real default rate per band (only when the file has loan_status)."""
    bands = [b for b in ui.DECISIONS if evaluation["rows_by_band"][b]]
    rates = [evaluation["default_rate_by_band"][b] for b in bands]
    fig = go.Figure(go.Bar(
        x=[ui.decision_label(b) for b in bands], y=rates,
        marker=dict(color=[ui.DECISIONS[b]["color"] for b in bands], cornerradius=6),
        text=[f"{r:.0%} of {evaluation['rows_by_band'][b]:,}" for b, r in zip(bands, rates)], textposition="outside",
        hovertemplate="%{x}: %{y:.1%} defaulted<extra></extra>",
    ))
    ui.style_figure(fig, "Real default rate per band (loan_status)", height=330)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15], title_text="Defaulted")
    return fig


def results_table(scored: list[dict]) -> pd.DataFrame:
    table = frame(scored)
    table["Probability of default"] = table["probability"].map(ui.format_probability)
    table["Decision"] = table["decision"].map(ui.decision_pill)
    names = {"row": "Row", "loan_id": "Loan ID", "applicant_name": "Applicant", "top_reason": "Top reason",
             "loan_status": "loan_status"}
    columns = [c for c in ["row", "loan_id", "applicant_name"] if c in table] + ["Probability of default", "Decision",
                                                                                 "top_reason"]
    columns += ["loan_status"] if "loan_status" in table else []
    return table[columns].rename(columns=names)


def show_result(result: dict) -> None:
    summary, scored, rejected = result["summary"], result["scored"], result["rejected"]
    st.subheader(f"Results - {result['file']}", anchor=False)
    kpi_tiles(summary)

    left, right = st.columns([2, 3])
    with left, st.container(key="card_batch_mix"):
        if summary["rows_scored"]:
            ui.show_chart(ui.decision_donut(summary["decisions"], "Decision mix", unit="rows"))
        else:
            st.info("No valid rows to score.")
    with right, st.container(key="card_batch_eval"):
        if summary["evaluation"]:
            ui.show_chart(default_rate_chart(summary["evaluation"]))
        else:
            cut = summary["cutoffs"]
            st.markdown('<div class="card-title">How the bands work</div>', unsafe_allow_html=True)
            st.markdown(f"APPROVE below **{cut['review']:.0%}** probability of default, REVIEW from "
                        f"**{cut['review']:.0%}** to **{cut['reject']:.0%}**, REJECT from **{cut['reject']:.0%}** "
                        f"(cut-offs from the model, version {summary['model_version']}).")
            st.info("Add a loan_status column (0 = repaid, 1 = defaulted) to see the ROC-AUC and the real "
                    "default rate per band.", icon="ℹ️")

    if scored:
        with st.container(key="card_batch_table"):
            st.caption(f"{len(scored):,} scored row(s). 'Row' is the row number in your file.")
            st.dataframe(results_table(scored), hide_index=True, width="stretch", key="batch_table",
                         column_config={"Decision": ui.decision_pill_column()})
            stem = Path(result["file"]).stem
            st.download_button("Download results CSV", data=frame(scored).to_csv(index=False),
                               file_name=f"{stem}_scored.csv", mime="text/csv", type="primary",
                               icon=":material/download:", key="batch_download")

    if rejected:
        with st.container(key="card_batch_rejected"):
            st.warning(f"{len(rejected):,} row(s) were not scored because they break a validation rule "
                       "(the same rules as the application form).", icon="⚠️")
            table = frame(rejected).rename(columns={"row": "Row", "loan_id": "Loan ID",
                                                           "applicant_name": "Applicant", "reason": "Reason"})
            st.dataframe(table, hide_index=True, width="stretch", key="batch_rejected_table")
            st.download_button("Download rejected rows CSV", data=frame(rejected).to_csv(index=False),
                               file_name=f"{Path(result['file']).stem}_rejected.csv", mime="text/csv",
                               icon=":material/download:", key="batch_download_rejected")


def page() -> None:
    ui.page_header("Batch scoring", "Upload a CSV of applicants (up to 5,000 rows). Every row is checked with the "
                   "same rules and scored by the same model as a single application. Nothing is stored.",
                   eyebrow="Bulk desk")

    upload, sample = st.columns([3, 2])
    with upload, st.container(key="card_batch_upload"):
        st.markdown('<div class="card-title">Your file</div>', unsafe_allow_html=True)
        file = st.file_uploader("CSV file", type=["csv"], key="batch_file", label_visibility="collapsed")
        st.caption(f"Required columns: {', '.join(REQUIRED)}. Optional: {', '.join(OPTIONAL)}.")
        if st.button("Score file", type="primary", icon=":material/bolt:", disabled=file is None, key="batch_score"):
            run(file.name, file.getvalue())
    with sample, st.container(key="card_batch_sample"):
        st.markdown('<div class="card-title">Need an example?</div>', unsafe_allow_html=True)
        st.caption("10 real loans from the test split, with names and the real outcome (loan_status).")
        st.download_button("Download sample CSV", data=SAMPLE_FILE.read_bytes(), file_name=SAMPLE_FILE.name,
                           mime="text/csv", icon=":material/download:", width="stretch", key="batch_sample_download")
        if st.button("Score the sample file", icon=":material/play_arrow:", width="stretch", key="batch_sample"):
            run(SAMPLE_FILE.name, SAMPLE_FILE.read_bytes())

    if st.session_state.get("batch_error"):
        st.error(st.session_state["batch_error"])
    if st.session_state.get("batch_result"):
        show_result(st.session_state["batch_result"])
