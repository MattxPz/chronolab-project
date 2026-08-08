"""Leaderboard: metricas, precision frente a coste, Diebold-Mariano y MCS."""

from __future__ import annotations

import streamlit as st

from chronolab.app.components import palette, state, widgets
from chronolab.viz.plots import plot_accuracy_vs_cost, plot_dm_heatmap

st.set_page_config(page_title="Chronolab · Leaderboard", page_icon="⏱️", layout="wide")

st.title("Leaderboard")
series = widgets.series_selector()

leaderboard = widgets.safe_load(state.load_leaderboard, name="el leaderboard")
if leaderboard is None:
    st.stop()

view = st.radio(
    "Vista",
    ["Agregado (todas las series)", f"Por serie: {series}" if series else "Por serie"],
    horizontal=True,
)
scope = (
    leaderboard.loc[leaderboard["unique_id"].isna()]
    if view.startswith("Agregado") or series is None
    else leaderboard.loc[leaderboard["unique_id"] == series]
)
scope = scope.sort_values("mase")

st.subheader("Metricas por modelo")
display_columns = [
    "model_id",
    "n_windows",
    "n_obs",
    "mae",
    "rmse",
    "mape",
    "smape",
    "mase",
    "coverage_95",
    "coverage_80",
    "coverage_50",
    "fit_seconds_total",
    "predict_seconds_total",
    "n_params",
    "is_zero_shot",
]
st.dataframe(
    scope[[c for c in display_columns if c in scope.columns]],
    width="stretch",
    hide_index=True,
)
st.caption("Encabezados ordenables. `mase < 1` bate al naive estacional de la propia serie.")

st.subheader("Precision frente a coste computacional")
cost_scope = leaderboard.loc[leaderboard["unique_id"].isna()].dropna(subset=["mase"])
st.plotly_chart(
    plot_accuracy_vs_cost(cost_scope, model_colors=palette.model_colors()),
    width="stretch",
)
st.caption(
    "Eje y: MASE (mas bajo, mejor). Eje x: coste total de entrenamiento en segundos, "
    "escala log. Agregado sobre todas las series."
)

st.subheader("Significancia de Diebold-Mariano")
dm_matrix = widgets.safe_load(state.load_dm_matrix, name="la matriz de Diebold-Mariano")
if dm_matrix is not None:
    st.plotly_chart(plot_dm_heatmap(dm_matrix), width="stretch")
    st.caption(
        "Estadistico DM entre pareja de modelos (fila vs columna): negativo y significativo "
        "(p < 0.05) significa que el modelo de la fila predice mejor que el de la columna. "
        "Calculado sobre los siete modelos con predicciones crudas del backtest demo, error "
        "absoluto pooled entre las tres series -una aproximacion, no la comparacion rigurosa "
        "que exige una unica serie por contraste."
    )
