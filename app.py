"""Streamlit dashboard: drift over two simulated years in production, and lineage for each forecast.

    streamlit run app.py        (on Windows: python -m streamlit run app.py)

Reads results/ and artifacts/lineage/. No competition data or model files are needed.
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.lineage.manifest import serving_version
from src.reporting.app_data import (
    REFERENCE_LABELS,
    STRATEGY_LABELS,
    drift_frame,
    load_lineage,
    load_results,
    versions_table,
)

st.set_page_config(page_title="Demand drift monitor", page_icon="📉", layout="wide")

try:
    DARK = st.context.theme.type == "dark"
except AttributeError:
    DARK = False
SERIES = ["#3987e5", "#d95926", "#199e70"] if DARK else ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED = "#898781"
TEXT = "#c3c2b7" if DARK else "#52514e"
STRATEGY_COLORS = {"drift_triggered": SERIES[0], "static": SERIES[1], "monthly": SERIES[2],
                   "static_with_recent_sales": MUTED}


@st.cache_data
def data():
    summary, monthly = load_results()
    return summary, monthly, load_lineage()


def legend(items: list[tuple[str, str]]) -> None:
    st.markdown(" &nbsp;&nbsp; ".join(
        f"<span style='white-space:nowrap'><span style='display:inline-block;width:14px;height:3px;"
        f"background:{color};vertical-align:middle;margin-right:6px'></span>{label}</span>"
        for label, color in items), unsafe_allow_html=True)


summary, monthly, lineage = data()
results = summary["summary"]
threshold = summary["config"]["drift_threshold"]
yearly = summary["data"]["yearly_mean_sales"]

st.title("Is the forecast still right?")
st.markdown(
    f"A demand forecast for **{summary['data']['stores']} stores × {summary['data']['items']} "
    "items**, trained on 2013–2015 and left running through 2016 and 2017 while average daily "
    f"sales grew from {yearly['2015']:.1f} to {yearly['2017']:.1f}. Every month an Evidently drift "
    "check compares the month with the model's training data, and every model version carries a "
    "lineage manifest naming the exact rows it learned from."
)

c1, c2, c3, c4 = st.columns(4)
bias_2017 = {s: results[s]["by_year"]["2017"]["bias_pct_mean"] for s in config.STRATEGIES}
c1.metric("Bias in 2017, never retrained", f"{bias_2017['static']:+.1f}%",
          help="Total forecast vs total actual sales. Negative = forecasting too little.")
c2.metric("Bias in 2017, retrain on drift", f"{bias_2017['drift_triggered']:+.1f}%")
c3.metric("Model versions: drift-triggered vs monthly",
          f"{results['drift_triggered']['model_versions']} vs "
          f"{results['monthly']['model_versions']}")
c4.metric("Forecasts traceable to training rows", "100%",
          help="Every logged forecast names its model version; every version has a manifest.")

tab_drift, tab_accuracy, tab_lineage, tab_explain = st.tabs(
    ["Drift over time", "Forecast accuracy", "Trace a forecast", "What is data drift?"])

# --- Drift ---------------------------------------------------------------------------------------
with tab_drift:
    left, right = st.columns(2)
    strategy = left.selectbox("Retraining strategy", config.STRATEGIES,
                              format_func=STRATEGY_LABELS.get)
    reference = right.radio("Compare each month with", list(REFERENCE_LABELS),
                            format_func=REFERENCE_LABELS.get, horizontal=True)
    frame = drift_frame(monthly, strategy, reference)
    legend([("Actual sales", SERIES[0]), ("Forecasts", SERIES[1])])
    x = alt.X("when:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=8))
    lines = alt.Chart(frame).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45)).encode(
        x=x, y=alt.Y("score:Q", title="Drift score (normalised Wasserstein)"),
        color=alt.Color("signal:N", legend=None,
                        scale=alt.Scale(domain=["Actual sales", "Forecasts"], range=SERIES[:2])),
        tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("signal:N", title="Column"),
                 alt.Tooltip("score:Q", title="Score", format=".3f"),
                 alt.Tooltip("drifted:N", title="Drift flagged")],
    )
    ref = pd.DataFrame({"y": [threshold], "t": [f"threshold {threshold}"]})
    rule = alt.Chart(ref).mark_rule(color=MUTED).encode(y="y:Q")
    rule_label = alt.Chart(ref).mark_text(align="left", dx=4, dy=-7, color=TEXT,
                                          fontSize=12).encode(x=alt.value(0), y="y:Q", text="t:N")
    retrained = monthly["retrained"].eq(True)  # blank for the comparison model
    retrains = monthly[(monthly["strategy"] == strategy) & retrained]
    marks = alt.Chart(retrains.assign(when=retrains["when"] + pd.Timedelta(days=27))).mark_rule(
        color=SERIES[2], strokeWidth=1.5, opacity=0.6).encode(
        x="when:T", tooltip=[alt.Tooltip("month:N", title="Retrained after")])
    drift_chart = alt.layer(marks, rule, rule_label, lines).properties(height=360)
    st.altair_chart(drift_chart, width="stretch")
    n_flags = int(frame[frame["signal"] == "Actual sales"]["drifted"].astype(bool).sum())
    st.caption(
        f"Green lines mark the end of a month after which the model was retrained. Sales drift was "
        f"flagged in {n_flags} of {frame['month'].nunique()} months with this comparison. "
        + ("The naive comparison flags seasons, not problems: July always looks different from "
           "the average of a whole year." if reference == "naive" else
           "Comparing like with like (July with July) leaves only real change: growth.")
    )

# --- Accuracy ------------------------------------------------------------------------------------
with tab_accuracy:
    shown = ["static", "drift_triggered", "monthly", "static_with_recent_sales"]
    legend([(STRATEGY_LABELS[s], STRATEGY_COLORS[s]) for s in shown])
    metric = st.radio("Measure", ["bias_pct", "smape"], horizontal=True,
                      format_func={"bias_pct": "Bias (%)", "smape": "SMAPE (%)"}.get)
    part = monthly[monthly["strategy"].isin(shown)]
    is_bias = metric == "bias_pct"
    chart = alt.Chart(part).mark_line(strokeWidth=2).encode(
        x=alt.X("when:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=8)),
        y=alt.Y(f"{metric}:Q", title="Bias, % of actual sales" if is_bias else "SMAPE, %"),
        color=alt.Color("strategy:N", legend=None,
                        scale=alt.Scale(domain=shown, range=[STRATEGY_COLORS[s] for s in shown])),
        tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("strategy:N", title="Strategy"),
                 alt.Tooltip(f"{metric}:Q", title="Value", format="+.2f" if is_bias else ".2f"),
                 alt.Tooltip("model_version:N", title="Model version")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED).encode(y="y:Q")
    st.altair_chart((zero + chart if metric == "bias_pct" else chart).properties(height=360),
                    width="stretch")
    table = pd.DataFrame([{
        "Strategy": STRATEGY_LABELS[s],
        "Model versions": results[s].get("model_versions", 1),
        "SMAPE 2016": results[s]["by_year"]["2016"]["smape_mean"],
        "SMAPE 2017": results[s]["by_year"]["2017"]["smape_mean"],
        "Bias 2016": results[s]["by_year"]["2016"]["bias_pct_mean"],
        "Bias 2017": results[s]["by_year"]["2017"]["bias_pct_mean"],
    } for s in shown])
    st.dataframe(table.style.format({"SMAPE 2016": "{:.2f}%", "SMAPE 2017": "{:.2f}%",
                                     "Bias 2016": "{:+.2f}%", "Bias 2017": "{:+.2f}%"}),
                 hide_index=True, width="stretch")

# --- Lineage -------------------------------------------------------------------------------------
with tab_lineage:
    st.markdown("Pick any forecast from the two production years and see which model made it "
                "and exactly what that model learned from.")
    a, b, c, d = st.columns(4)
    trace_strategy = a.selectbox("Strategy", list(lineage),
                                 index=list(lineage).index("drift_triggered"),
                                 format_func=STRATEGY_LABELS.get, key="trace_strategy")
    day = b.date_input("Forecast date", value=pd.Timestamp("2017-03-14"),
                       min_value=pd.Timestamp(config.PRODUCTION_START),
                       max_value=pd.Timestamp(config.PRODUCTION_END))
    store = c.number_input("Store", 1, summary["data"]["stores"], 3)
    item = d.number_input("Item", 1, summary["data"]["items"], 17)
    manifest = serving_version(lineage[trace_strategy], day)
    data_info, trigger = manifest["training_data"], manifest["trigger"]
    if trigger["type"] == "initial":
        why = "the initial model"
    elif trigger["type"] == "drift":
        why = (f"retrained after {trigger['month']} because sales drifted "
               f"(score {trigger['score']:.3f} ≥ {trigger['threshold']} "
               f"vs {trigger['reference_month']})")
    else:
        why = f"retrained after {trigger['month']} on schedule"
    st.success(
        f"The forecast for store {store}, item {item} on {day:%d %b %Y} was made by "
        f"**{manifest['model_version']}**, {why}. It was trained on "
        f"**{data_info['date_from']} to {data_info['date_to']}** ({data_info['rows']:,} rows) and "
        f"served from {manifest['served']['from']} to {manifest['served']['to']}."
    )
    st.code(f"training snapshot sha256  {data_info['snapshot_sha256']}\n"
            f"source file sha256        {data_info['source_sha256']}\n"
            f"model file sha256         {manifest['model']['file_sha256']}\n"
            f"code sha256               {manifest['code_sha256']}", language=None)
    st.markdown("Verify it yourself with the data downloaded: "
                f"`python -m src.lineage.trace --date {day:%Y-%m-%d} --store {store} --item {item} "
                f"--strategy {trace_strategy}` recomputes the snapshot fingerprint from "
                "`data/train.csv` and reports VERIFIED or MISMATCH.")
    with st.expander("Full manifest"):
        st.code(json.dumps(manifest, indent=2), language="json")
    st.subheader("All versions for this strategy")
    st.dataframe(versions_table(lineage[trace_strategy]), hide_index=True, width="stretch")

# --- Explanation ---------------------------------------------------------------------------------
with tab_explain:
    st.markdown(
        f"""
**Data drift** is when the data a model sees in production stops looking like the data it
learned from. A model left running unattended doesn't complain when that happens. It keeps
producing forecasts with the same confidence, and nobody finds out until stock runs short.

**What happened here.** Average daily sales grew every year (from {yearly['2015']:.1f} in 2015
to {yearly['2017']:.1f} in 2017). The forecast only knows store, item and calendar, so it
kept predicting 2015 levels. Its errors looked ordinary, but its forecasts were consistently
too low: {results['static']['by_year']['2016']['bias_pct_mean']:+.1f}% in 2016 and
{results['static']['by_year']['2017']['bias_pct_mean']:+.1f}% in 2017.

**Two lessons from the monitor:**

- **Compare like with like.** Sales are seasonal, so July always differs from a whole-year
  average. A naive check raises alarms every summer and winter. Comparing each month with the
  same month in the training data removes those false alarms and leaves the real change.
- **Drift isn't the same as decay.** A second model that also sees last month's sales went through
  the same growth. Its inputs and forecasts drifted along with sales, and its bias stayed at
  {results['static_with_recent_sales']['bias_pct_mean']:+.1f}%. The warning sign is sales drifting
  while the forecasts don't follow.

**Lineage** answers the question every drift alarm raises: which model made this number, and
what did it learn from? Each version's manifest records the training window and a fingerprint
of those exact rows, so the answer can be checked, not just asserted.
"""
    )

st.caption(f"Results generated {summary['generated_at']} · "
           f"Evidently {summary['versions']['evidently']} · "
           f"XGBoost {summary['versions']['xgboost']}")
