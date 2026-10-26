"""Dashboard (admin only): usage, latency, decision mix, model info and drift status."""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from api import ApiClient, ApiError

LATENCY_TARGET_MS = 200     # project target for one scoring request


def whole_number_axis(fig: go.Figure, largest: int) -> None:
    """Counts are whole numbers: no 0.5 ticks on the y axis."""
    fig.update_yaxes(rangemode="tozero", dtick=max(1, math.ceil(largest / 5)))


def ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} ms"


def kpi_tiles(stats: dict, info: dict, days: pd.DataFrame) -> None:
    c = st.columns(4)
    with c[0], st.container(key="kpi_scored"):
        st.metric("Applications scored", f"{stats['total_requests']:,}",
                  delta=f"{int(days['requests'].sum()):,} in the last 7 days", delta_color="off", delta_arrow="off")
    target = dict(delta=f"target < {LATENCY_TARGET_MS} ms", delta_color="off", delta_arrow="off")
    with c[1], st.container(key="kpi_latency"):
        st.metric("Average latency", ms(stats["avg_latency_ms"]), **target)
    with c[2], st.container(key="kpi_p95"):
        st.metric("95% of requests faster than", ms(stats["p95_latency_ms"]), **target)
    with c[3], st.container(key="kpi_auc"):
        st.metric("Test ROC-AUC", f"{info['test_metrics']['roc_auc']:.3f}", delta="HCL target >= 0.85",
                  delta_color="off", delta_arrow="off", help="On the held-out test split")


def decision_donut(mix: dict) -> go.Figure:
    total = sum(mix.values())
    mix = {d: n for d, n in mix.items() if n}          # no "0%" slice labels
    fig = go.Figure(go.Pie(
        labels=[ui.decision_label(d) for d in mix], values=list(mix.values()), hole=0.62, sort=False,
        marker=dict(colors=[ui.DECISIONS[d]["color"] for d in mix], line=dict(color=ui.WHITE, width=3)),
        textinfo="percent", insidetextorientation="horizontal",
        textfont=dict(color=[ui.DECISIONS[d]["ink"] for d in mix]),
        hovertemplate="%{label}: %{value:,} applications<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{total:,}</b><br>applications", showarrow=False, font=dict(size=15))
    return ui.style_figure(fig, "Decision mix (all time)", height=330)


def per_day_chart(days: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for decision in ui.DECISIONS:
        fig.add_bar(x=days["date"], y=days[decision.lower()], name=ui.decision_label(decision),
                    marker=dict(color=ui.DECISIONS[decision]["color"], cornerradius=4))
    ui.style_figure(fig, "Applications per day (last 7 days)", height=330)
    fig.update_layout(barmode="stack", yaxis_title="Applications", legend=dict(traceorder="normal"))
    whole_number_axis(fig, int(days["requests"].max()))
    return fig


def latency_chart(stats: dict) -> go.Figure:
    names, values = ["Average", "95th percentile"], [stats["avg_latency_ms"], stats["p95_latency_ms"]]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker=dict(color=[ui.AUBERGINE, ui.MARIGOLD], cornerradius=6),
        text=[ms(v) for v in values], textposition="outside", hovertemplate="%{y}: %{x:.0f} ms<extra></extra>",
    ))
    fig.add_vline(x=LATENCY_TARGET_MS, line_dash="dot", line_color=ui.CRIMSON,
                  annotation_text=f"target {LATENCY_TARGET_MS} ms", annotation_position="top")
    ui.style_figure(fig, "Latency per request", height=230)
    fig.update_xaxes(title_text="milliseconds", range=[0, max(LATENCY_TARGET_MS, *values) * 1.25])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def model_card(info: dict) -> None:
    t, cut = info["test_metrics"], info["decision_cutoffs"]
    rows = [
        ("Model", f"version {info['version']}"), ("Type", info["model_type"]),
        ("ROC-AUC / PR-AUC", f"{t['roc_auc']:.3f} / {t['pr_auc']:.3f} (test)"),
        ("Calibration error", f"{t['ece']:.3f}"), ("Trained", f"{info['training_date'][:10]} on "
                                                          f"{info['n_train_val']:,} loans"),
    ]
    items = "".join(f"<dt>{label}</dt><dd>{ui.safe_html(value)}</dd>" for label, value in rows)
    bands = (f'<span class="band" style="--band:{ui.GREEN}; --ink:{ui.WHITE}">APPROVE &lt; {cut["review"]:.0%}</span>'
             f'<span class="band" style="--band:{ui.MARIGOLD}; --ink:{ui.AUBERGINE}">REVIEW {cut["review"]:.0%}'
             f' - {cut["reject"]:.0%}</span>'
             f'<span class="band" style="--band:{ui.CRIMSON}; --ink:{ui.WHITE}">REJECT &ge; {cut["reject"]:.0%}</span>')
    st.markdown(f'<div class="card-title">Serving model</div><dl class="info-list">{items}</dl>'
                f'<div class="bands">{bands}</div>', unsafe_allow_html=True)


def drift_card(client: ApiClient, drift: dict | None) -> None:
    status = "none" if drift is None else ("alert" if drift["status"] == "DRIFT DETECTED" else "ok")
    with st.container(key=f"drift_card_{status}"):
        st.subheader(":material/radar: Data drift", anchor=False)
        if drift is None:
            st.info("No drift check has run yet.", icon="ℹ️")
        else:
            text = (f"**{drift['status']}** - {drift['batch_name']} checked on {ui.format_time(drift['created_at'])}: "
                    f"{drift['drift_share']:.0%} of features drifted, prediction PSI {drift['prediction_psi']:.3f}")
            (st.error if status == "alert" else st.success)(text, icon="🚨" if status == "alert" else "✅")
        st.caption("A drift check compares new applications with the training data. It runs every day at 02:00 on "
                   "the applications of the last 30 days (needs at least 50); the demo batches are simulated from "
                   "the test split.")

        st.markdown("**Run a drift check now**")
        batch = st.radio("Data to check", ["live", "no_drift", "drift"], horizontal=True, key="drift_batch",
                         format_func={"live": "Live applications (last 30 days)",
                                      "no_drift": "Demo: batch without drift", "drift": "Demo: batch with drift"}.get)
        if st.button("Run drift check", key="run_drift", type="primary", icon=":material/play_arrow:"):
            with st.spinner("Comparing with the training data..."):
                try:
                    result = client.run_drift(batch)
                except ApiError as e:
                    st.error(e.message)
                    return
            if result["status"] == "SKIPPED":
                st.warning(result["message"], icon="⏭️")
            else:
                st.rerun()   # show the new result at the top of this card


def page() -> None:
    ui.page_header("Dashboard", "How the model is used and whether new applications still look like the "
                   "training data.", eyebrow="Admin")
    client = ApiClient(st.session_state["token"])
    try:
        info, stats, drift = client.model_info(), client.stats(), client.latest_drift()
    except ApiError as e:
        st.error(e.message)
        return

    days = pd.DataFrame(stats["last_7_days"])
    kpi_tiles(stats, info, days)

    left, right = st.columns([2, 3])
    with left, st.container(key="card_mix"):
        if sum(stats["decisions"].values()):
            ui.show_chart(decision_donut(stats["decisions"]))
        else:
            st.info("No applications scored yet.")
    with right, st.container(key="card_per_day"):
        ui.show_chart(per_day_chart(days))

    left, right = st.columns([1, 1])
    with left, st.container(key="card_latency"):
        if stats["avg_latency_ms"] is None:
            st.info("No latency measured yet.")
        else:
            ui.show_chart(latency_chart(stats))
    with right, st.container(key="card_model"):
        model_card(info)

    drift_card(client, drift)
