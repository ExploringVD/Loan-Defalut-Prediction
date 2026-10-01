"""Dashboard (admin only): model info, usage, latency, decision mix and drift status."""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from api import ApiClient, ApiError


def whole_number_axis(fig: go.Figure, largest: int) -> None:
    """Counts are whole numbers: no 0.5 ticks on the y axis."""
    fig.update_yaxes(rangemode="tozero", dtick=max(1, math.ceil(largest / 5)))


def page() -> None:
    st.title("Dashboard")
    client = ApiClient(st.session_state["token"])
    try:
        info, stats, drift = client.model_info(), client.stats(), client.latest_drift()
    except ApiError as e:
        st.error(e.message)
        return

    st.subheader("Model")
    t, cut = info["test_metrics"], info["decision_cutoffs"]
    c = st.columns(5)
    c[0].metric("Model version", info["version"])
    c[1].metric("Test ROC-AUC", f"{t['roc_auc']:.3f}")
    c[2].metric("Test PR-AUC", f"{t['pr_auc']:.3f}")
    c[3].metric("Calibration error", f"{t['ece']:.3f}")
    c[4].metric("Trained", info["training_date"][:10])
    st.caption(f"{info['model_type']} - APPROVE below {cut['review']:.0%}, REVIEW {cut['review']:.0%} to "
               f"{cut['reject']:.0%}, REJECT from {cut['reject']:.0%} probability of default")

    st.subheader("Usage")
    c = st.columns(3)
    c[0].metric("Applications scored", f"{stats['total_requests']:,}")
    c[1].metric("Average latency", "-" if stats["avg_latency_ms"] is None else f"{stats['avg_latency_ms']:.0f} ms")
    c[2].metric("95% of requests faster than", "-" if stats["p95_latency_ms"] is None else f"{stats['p95_latency_ms']:.0f} ms")

    left, right = st.columns([3, 2])
    days = pd.DataFrame(stats["last_7_days"])
    per_day = go.Figure()
    for decision in ["APPROVE", "REVIEW", "REJECT"]:
        per_day.add_bar(x=days["date"], y=days[decision.lower()], name=ui.decision_label(decision),
                        marker_color=ui.DECISIONS[decision]["color"])
    per_day.update_layout(barmode="stack", title="Applications per day (last 7 days)", yaxis_title="Applications",
                          height=340, margin=dict(l=10, r=10, t=50, b=40),
                          legend=dict(orientation="h", y=-0.2, traceorder="normal"))
    whole_number_axis(per_day, int(days["requests"].max()))
    left.plotly_chart(per_day, width="stretch")

    mix = stats["decisions"]
    total = sum(mix.values()) or 1
    decision_mix = go.Figure(go.Bar(
        x=[ui.decision_label(d) for d in mix], y=list(mix.values()),
        marker_color=[ui.DECISIONS[d]["color"] for d in mix],
        text=[f"{n:,} ({n / total:.0%})" for n in mix.values()], textposition="outside",
    ))
    decision_mix.update_layout(title="Decision mix (all time)", yaxis_title="Applications", height=340,
                               margin=dict(l=10, r=10, t=50, b=40))
    whole_number_axis(decision_mix, max(mix.values()))
    decision_mix.update_yaxes(range=[0, max(mix.values()) * 1.2 + 1])   # room for the labels above the bars
    right.plotly_chart(decision_mix, width="stretch")

    st.subheader("Data drift")
    st.caption("A drift check compares new applications with the training data. It runs every day at 02:00 on the "
               "applications of the last 30 days (needs at least 50); the demo batches are simulated from the test split.")
    if drift is None:
        st.info("No drift check has run yet.", icon="ℹ️")
    else:
        text = (f"**{drift['status']}** - {drift['batch_name']} checked on {ui.format_time(drift['created_at'])}: "
                f"{drift['drift_share']:.0%} of features drifted, prediction PSI {drift['prediction_psi']:.3f}")
        (st.error if drift["status"] == "DRIFT DETECTED" else st.success)(
            text, icon="🚨" if drift["status"] == "DRIFT DETECTED" else "✅")

    with st.expander("Run a drift check now"):
        batch = st.radio("Data to check", ["live", "no_drift", "drift"], horizontal=True, key="drift_batch",
                         format_func={"live": "Live applications (last 30 days)",
                                      "no_drift": "Demo: batch without drift", "drift": "Demo: batch with drift"}.get)
        if st.button("Run drift check", key="run_drift"):
            with st.spinner("Comparing with the training data..."):
                try:
                    result = client.run_drift(batch)
                except ApiError as e:
                    st.error(e.message)
                    return
            if result["status"] == "SKIPPED":
                st.warning(result["message"], icon="⏭️")
            else:
                st.rerun()   # show the new result at the top of this section
