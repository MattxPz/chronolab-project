"""Anomalias: series marcadas, slider de alfa, comparativa y tabla de eventos."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chronolab.app.components import palette, state, widgets
from chronolab.viz.plots import anomaly_threshold, plot_anomaly_series

st.set_page_config(page_title="Chronolab · Anomalias", page_icon="⏱️", layout="wide")

st.title("Anomalias")
series = widgets.series_selector()
detector = widgets.detector_selector()
alpha = widgets.alpha_slider()

if series is None or detector is None:
    widgets.missing_artifact(
        "los scores de anomalias",
        detail="Genera `anomaly_scores.parquet` con `scripts/run_anomaly_eval.py`.",
    )
    st.stop()

panel = widgets.safe_load(state.load_panel, name="la serie demo")
scores = widgets.safe_load(state.load_anomaly_scores, name="los scores de anomalias")
truth = widgets.safe_load(state.load_anomaly_truth, name="los eventos anomalos (ground truth)")
if panel is None or scores is None or truth is None:
    st.stop()

threshold = anomaly_threshold(alpha)
st.sidebar.caption(f"Umbral vigente: score ≥ {threshold:.2f}  (-log10({alpha:.3f}))")

series_frame = panel.loc[panel["unique_id"] == series, ["ds", "y"]]
scores_frame = scores.loc[(scores["unique_id"] == series) & (scores["detector_id"] == detector)]
truth_frame = truth.loc[truth["unique_id"] == series]

events = pd.DataFrame()
if not truth_frame.empty:
    events = (
        truth_frame.groupby("event_id")
        .agg(
            anomaly_type=("anomaly_type", "first"),
            severity=("severity", "max"),
            start=("ds", "min"),
            end=("ds", "max"),
            n_points=("ds", "count"),
        )
        .reset_index()
    )

    def _detection(row: pd.Series) -> pd.Series:
        span = scores_frame.loc[
            (scores_frame["ds"] >= row["start"]) & (scores_frame["ds"] <= row["end"])
        ]
        if span.empty:
            return pd.Series({"max_score": float("nan"), "detectado": False})
        detected = bool((span["scorable"].fillna(False) & (span["score"] >= threshold)).any())
        return pd.Series({"max_score": float(span["score"].max()), "detectado": detected})

    events = events.join(events.apply(_detection, axis=1)).sort_values("severity", ascending=False)

st.subheader(f"{series} — {detector}")
zoom_labels = ["Vista completa"] + [
    f"{row.anomaly_type} · severidad {row.severity:.1f} · {row.start:%m-%d %Hh}"
    for row in events.itertuples()
]
zoom_choice = st.selectbox("Zoom a evento", zoom_labels)

fig = plot_anomaly_series(
    series_frame,
    scores_frame,
    truth_frame,
    threshold=threshold,
    color=palette.series_colors().get(series, "#2a78d6"),
    unique_id=series,
)
fig = widgets.with_rangeslider(fig)
if zoom_choice != "Vista completa":
    chosen = events.iloc[zoom_labels.index(zoom_choice) - 1]
    pad = pd.Timedelta(hours=12)
    fig.update_xaxes(range=[chosen["start"] - pad, chosen["end"] + pad])
st.plotly_chart(fig, width="stretch")

st.subheader("Eventos, ordenados por severidad")
if events.empty:
    st.info("Esta serie no tiene eventos anomalos inyectados.", icon="💡")
else:
    st.dataframe(
        events[["anomaly_type", "severity", "start", "end", "n_points", "max_score", "detectado"]],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "`detectado` se recalcula con el umbral vigente del slider de alfa, sobre los scores "
        "ya calculados por el detector -no vuelve a puntuar nada."
    )
