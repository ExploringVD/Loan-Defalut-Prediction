"""History: past applications with filters (decision, dates), a count per decision and a detail view."""
import pandas as pd
import streamlit as st

import ui
from api import ApiClient, ApiError

PAGE_SIZE = 50


def decision_counts(client: ApiClient, date_from, date_to) -> dict:
    """Applications per decision for the chosen dates (only the totals are read: page_size=1)."""
    counts = {"All": client.list_applications(page=1, page_size=1, date_from=date_from, date_to=date_to)["total"]}
    for d in ui.DECISIONS:
        counts[d] = client.list_applications(page=1, page_size=1, decision=d, date_from=date_from,
                                             date_to=date_to)["total"]
    return counts


def kpi_strip(counts: dict) -> None:
    tiles = [("All", "Total applications"), ("APPROVE", "Approved"), ("REVIEW", "To review"), ("REJECT", "Rejected")]
    for col, (key, label) in zip(st.columns(4), tiles):
        with col, st.container(key=f"kpi_{key.lower()}"):
            if key == "All":
                note = "for the chosen dates"
            else:
                note = f"{counts[key] / counts['All']:.0%} of all" if counts["All"] else "-"
            st.metric(label, f"{counts[key]:,}", delta=note, delta_color="off", delta_arrow="off")


def page() -> None:
    ui.page_header("Application history", "Every scored application with its decision. Filter by decision or "
                   "date and click a row for the full explanation.", eyebrow="Records")
    client = ApiClient(st.session_state["token"])

    with st.container(key="card_filters"):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 0.8])
        decision = c1.selectbox("Decision", ["All", "APPROVE", "REVIEW", "REJECT"], key="history_decision")
        date_from = c2.date_input("From", value=None, key="history_from", help="Created on or after (UTC)")
        date_to = c3.date_input("To", value=None, key="history_to", help="Created on or before (UTC)")
        page_no = c4.number_input("Page", min_value=1, value=1, step=1, key="history_page")

    try:
        kpi_strip(decision_counts(client, date_from, date_to))
        result = client.list_applications(page=int(page_no), page_size=PAGE_SIZE,
                                          decision=None if decision == "All" else decision,
                                          date_from=date_from, date_to=date_to)
    except ApiError as e:
        st.error(e.message)
        return

    items = result["items"]
    pages = max(1, -(-result["total"] // PAGE_SIZE))
    if not items:
        st.info("No applications match these filters.")
        return

    table = pd.DataFrame([{
        "ID": a["id"],
        "Created": ui.format_time(a["created_at"]),
        "Applicant": a["applicant_name"],
        "Loan amount": a["loan_amnt"],
        "Grade": a["loan_grade"],
        "Probability of default": ui.format_probability(a["prediction"]["probability"]) if a["prediction"] else "-",
        # a list, because the pill column (MultiselectColumn) shows each list item as a coloured pill
        "Decision": [ui.decision_label(a["prediction"]["decision"])] if a["prediction"] else [],
        "Entered by": a["created_by"],
    } for a in items])
    with st.container(key="card_table"):
        st.caption(f"{result['total']:,} application(s) - page {result['page']} of {pages} - click a row to see "
                   "the details.")
        selection = st.dataframe(
            table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            key="history_table",
            column_config={
                "Loan amount": st.column_config.NumberColumn(format="$%d"),
                "Decision": st.column_config.MultiselectColumn(
                    options=[ui.decision_label(d) for d in ui.DECISIONS],
                    color=[v["pill"] for v in ui.DECISIONS.values()]),
            })

    rows = selection.selection.rows if selection else []
    if rows:
        chosen = items[rows[0]]
        try:
            detail = client.get_application(chosen["id"])
        except ApiError as e:
            st.error(e.message)
            return
        st.subheader(f"Application #{chosen['id']} - {chosen['applicant_name']}", anchor=False)
        left, right = st.columns([1, 2.5])
        with left, st.container(key="card_inputs"):
            inputs = ui.applicant_table(detail)
            st.dataframe(inputs, hide_index=True, width="stretch", height=35 * (len(inputs) + 1) + 3)   # no scrolling
        with right:
            ui.show_result(detail)
