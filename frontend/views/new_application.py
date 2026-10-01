"""New Application: the 11 fields in three groups, scored by POST /applications."""
import streamlit as st

import ui
from api import ApiClient, ApiError

# Limits must match the API (app/schemas.py) - tests/test_frontend.py checks this against the API's OpenAPI schema
LIMITS = {
    "person_age": (18, 100), "person_income": (1.0, 10_000_000.0), "person_emp_length": (0.0, 86.0),
    "loan_amnt": (1.0, 100_000.0), "loan_int_rate": (0.1, 40.0), "cb_person_cred_hist_length": (0, 100),
}
DEFAULTS = {   # a typical applicant (medians of the data)
    "applicant_name": "", "person_age": 26, "person_income": 55_000.0, "person_emp_length": 4.0,
    "person_home_ownership": "RENT", "loan_intent": "EDUCATION", "loan_amnt": 8_000.0, "loan_grade": "B",
    "loan_int_rate": 11.0, "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 4,
}


def field_error(name: str) -> None:
    """Show the API's 422 message for this field right under it."""
    message = st.session_state.get("form_errors", {}).get(name)
    if message:
        st.markdown(f":red[⚠ {message}]")


def number(label, key, help_text, step, fmt=None, optional=False):
    """Number box with the API's limits. Optional fields get a 'Not known' tick box (sent as null)."""
    lo, hi = LIMITS[key]
    unknown = st.checkbox(f"{label.split(' (')[0]}: not known", key=f"{key}_unknown") if optional else False
    value = st.number_input(label, min_value=lo, max_value=hi, value=DEFAULTS[key], step=step, format=fmt,
                            key=key, help=help_text, disabled=unknown)
    field_error(key)
    return None if unknown else value


def page() -> None:
    st.title("New loan application")
    st.caption("Fill in the applicant's details and press **Score application**. For the two optional fields you "
               "can tick *not known* - the model then fills in a typical value.")

    st.text_input("Applicant name", value=DEFAULTS["applicant_name"], key="applicant_name", max_chars=100,
                  help="Used only to find the application later")
    field_error("applicant_name")
    values = {}

    col1, col2, col3 = st.columns(3)
    with col1, st.container(border=True):
        st.subheader("Applicant")
        values["person_age"] = number("Age (years)", "person_age", "18 to 100", step=1)
        values["person_income"] = number("Yearly income ($)", "person_income", "Gross yearly income in dollars",
                                         step=1_000.0, fmt="%.0f")
        values["person_emp_length"] = number("Years employed (optional)", "person_emp_length",
                                             "Years in the current job. Cannot be more than age - 14.",
                                             step=1.0, fmt="%.0f", optional=True)
        values["person_home_ownership"] = st.selectbox(
            "Home ownership", list(ui.HOME_LABELS), index=list(ui.HOME_LABELS).index(DEFAULTS["person_home_ownership"]),
            format_func=ui.HOME_LABELS.get, key="person_home_ownership")
        field_error("person_home_ownership")

    with col2, st.container(border=True):
        st.subheader("Loan")
        values["loan_intent"] = st.selectbox(
            "Purpose", list(ui.INTENT_LABELS), index=list(ui.INTENT_LABELS).index(DEFAULTS["loan_intent"]),
            format_func=ui.INTENT_LABELS.get, key="loan_intent")
        field_error("loan_intent")
        values["loan_amnt"] = number("Loan amount ($)", "loan_amnt", "Amount requested in dollars", step=500.0, fmt="%.0f")
        values["loan_grade"] = st.select_slider("Loan grade", options=list("ABCDEFG"), value=DEFAULTS["loan_grade"],
                                                key="loan_grade", help="The lender's risk grade: A = safest, G = riskiest")
        field_error("loan_grade")
        values["loan_int_rate"] = number("Interest rate % (optional)", "loan_int_rate", "Yearly interest rate in percent",
                                         step=0.1, fmt="%.2f", optional=True)
        income, amount = values["person_income"], values["loan_amnt"]
        ratio = round(amount / income, 2) if income and amount else 0.0
        st.text_input("Loan as % of yearly income", value=f"{ratio:.0%}", disabled=True,
                      help="Calculated: loan amount / yearly income (sent to the API as loan_percent_income)")
        if ratio > 1:
            st.markdown(":red[⚠ The loan is larger than the yearly income - the API only accepts up to 100%.]")
        field_error("loan_percent_income")

    with col3, st.container(border=True):
        st.subheader("Credit history")
        values["cb_person_default_on_file"] = st.radio(
            "Past default on file?", ["N", "Y"], format_func={"N": "No", "Y": "Yes"}.get, horizontal=True,
            key="cb_person_default_on_file", help="From the credit bureau: has the applicant defaulted before?")
        field_error("cb_person_default_on_file")
        values["cb_person_cred_hist_length"] = number(
            "Credit history length (years)", "cb_person_cred_hist_length",
            "Years since the first credit account. Cannot be more than the age.", step=1)

    if st.button("Score application", type="primary"):
        body = {"applicant_name": st.session_state["applicant_name"].strip(), **values, "loan_percent_income": ratio}
        st.session_state["form_errors"] = {}
        try:
            st.session_state["last_result"] = ApiClient(st.session_state["token"]).create_application(body)
        except ApiError as e:
            st.session_state["form_errors"] = e.field_errors
            st.session_state["last_result"] = None
            st.session_state["form_message"] = (e.message, e.general_errors)
        else:
            st.session_state["form_message"] = None
        st.rerun()   # redraw so the field errors appear next to the fields

    message = st.session_state.get("form_message")
    if message:
        st.error(message[0])
        for extra in message[1]:
            st.error(extra)
    if st.session_state.get("last_result"):
        st.divider()
        ui.show_result(st.session_state["last_result"])
