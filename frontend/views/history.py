"""History: past applications with filters (decision, dates) and a detail view."""
import pandas as pd
import streamlit as st

import ui
from api import ApiClient, ApiError

PAGE_SIZE = 50


def page() -> None:
    st.title("Application history")
    client = ApiClient(st.session_state["token"])

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 0.8])
    decision = c1.selectbox("Decision", ["All", "APPROVE", "REVIEW", "REJECT"], key="history_decision")
    date_from = c2.date_input("From", value=None, key="history_from", help="Created on or after (UTC)")
    date_to = c3.date_input("To", value=None, key="history_to", help="Created on or before (UTC)")
    page_no = c4.number_input("Page", min_value=1, value=1, step=1, key="history_page")

    try:
        result = client.list_applications(page=int(page_no), page_size=PAGE_SIZE,
                                          decision=None if decision == "All" else decision,
                                          date_from=date_from, date_to=date_to)
    except ApiError as e:
        st.error(e.message)
        return

    items = result["items"]
    pages = max(1, -(-result["total"] // PAGE_SIZE))
    st.caption(f"{result['total']:,} application(s) - page {result['page']} of {pages}")
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
        "Decision": ui.decision_label(a["prediction"]["decision"]) if a["prediction"] else "-",
        "Entered by": a["created_by"],
    } for a in items])
    st.caption("Click a row to see the details.")
    selection = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key="history_table",
                             column_config={"Loan amount": st.column_config.NumberColumn(format="$%d")})

    rows = selection.selection.rows if selection else []
    if rows:
        chosen = items[rows[0]]
        st.divider()
        st.subheader(f"Application #{chosen['id']} - {chosen['applicant_name']}")
        try:
            detail = client.get_application(chosen["id"])
        except ApiError as e:
            st.error(e.message)
            return
        left, right = st.columns([1, 2.5])
        inputs = ui.applicant_table(detail)
        left.dataframe(inputs, hide_index=True, width="stretch", height=35 * (len(inputs) + 1) + 3)   # no scrolling
        with right:
            ui.show_result(detail)
