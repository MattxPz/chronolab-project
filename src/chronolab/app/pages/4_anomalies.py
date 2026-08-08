"""Anomalias: series marcadas, slider de alfa, comparativa y tabla de eventos."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chronolab.app.components import palette, state, theme, widgets
from chronolab.viz.plots import anomaly_threshold, plot_anomaly_series

st.set_page_config(page_title="Chronolab · Anomalias", page_icon="⏱️", layout="wide")
theme.apply_density(page_title="anomalies")

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
results = widgets.safe_load(state.load_anomaly_results, name="las metricas de deteccion")
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
zoom_key = "chronolab_anomaly_zoom"
if st.session_state.get(zoom_key) not in zoom_labels:
    st.session_state[zoom_key] = zoom_labels[0]
zoom_choice = st.selectbox("Zoom a evento", zoom_labels, key=zoom_key)

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

st.markdown("##### En resumen, con el umbral vigente")
cards = st.columns(4)
with cards[0], st.container(border=True):
    if events.empty:
        st.metric("Eventos detectados", "—")
    else:
        st.metric("Eventos detectados", f"{int(events['detectado'].sum())}/{len(events)}")
with cards[1], st.container(border=True):
    flagged = int(
        (scores_frame["scorable"].fillna(False) & (scores_frame["score"] >= threshold)).sum()
    )
    st.metric(
        "Puntos marcados",
        flagged,
        help="Instantes con score >= umbral, dentro o fuera de un evento real.",
    )

if results is not None:
    pooled = results.loc[
        (results["detector_id"] == detector)
        & (results["unique_id"] == series)
        & (results["anomaly_type"] == "all")
    ]

    def _metric(name: str) -> float | None:
        row = pooled.loc[pooled["metric"] == name]
        return float(row["value"].iloc[0]) if not row.empty else None

    range_recall, auc_pr, auc_pr_baseline = (
        _metric("range_recall"),
        _metric("auc_pr"),
        _metric("auc_pr_baseline"),
    )
    with cards[2], st.container(border=True):
        st.metric(
            "Range recall",
            "—" if range_recall is None else f"{range_recall:.2f}",
            help=theme.METRIC_HELP["range_recall"]
            + " Calculado a alfa=0.05 (calibracion original), no se recalcula con el slider.",
        )
    with cards[3], st.container(border=True):
        delta = None if auc_pr is None or auc_pr_baseline is None else auc_pr - auc_pr_baseline
        st.metric(
            "AUC-PR",
            "—" if auc_pr is None else f"{auc_pr:.2f}",
            delta=None if delta is None else f"{delta:+.2f} vs prevalencia",
            help=theme.METRIC_HELP["auc_pr"],
        )

st.subheader("Eventos, ordenados por severidad")
if events.empty:
    st.info("Esta serie no tiene eventos anomalos inyectados.", icon="💡")
else:
    st.dataframe(
        events[["anomaly_type", "severity", "start", "end", "n_points", "max_score", "detectado"]],
        width="stretch",
        hide_index=True,
        column_config={
            "anomaly_type": st.column_config.TextColumn("Tipo"),
            "severity": st.column_config.NumberColumn(
                "Severidad",
                help="Desviaciones tipicas locales del desplazamiento inyectado.",
                format="%.1f",
            ),
            "start": st.column_config.DatetimeColumn("Inicio"),
            "end": st.column_config.DatetimeColumn("Fin"),
            "n_points": st.column_config.NumberColumn("Puntos"),
            "max_score": st.column_config.NumberColumn(
                "Score maximo",
                help="Score mas alto del detector dentro del rango del evento.",
                format="%.2f",
            ),
            "detectado": st.column_config.CheckboxColumn(
                "Detectado",
                help="Con el umbral vigente del slider de alfa, sobre scores ya calculados.",
            ),
        },
    )
    st.caption(
        "`Detectado` se recalcula con el umbral vigente del slider de alfa, sobre los scores "
        "ya calculados por el detector — no vuelve a puntuar nada."
    )

theme.footer()
