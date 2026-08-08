"""Forecast: predicciones superpuestas con bandas, por serie, modelo y horizonte."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chronolab.app.components import palette, state, widgets
from chronolab.viz.plots import plot_forecast_overlay, plot_residuals

st.set_page_config(page_title="Chronolab · Forecast", page_icon="⏱️", layout="wide")

st.title("Forecast")
series = widgets.series_selector()
models = widgets.model_multiselect(label="Modelos", default_n=3)

if series is None:
    widgets.missing_artifact("la serie demo")
    st.stop()

forecasts_all = widgets.safe_load(state.load_forecasts, name="las predicciones del backtest demo")
windows = widgets.safe_load(state.load_windows, name="las ventanas del backtest demo")
panel = widgets.safe_load(state.load_panel, name="la serie demo")
if forecasts_all is None or windows is None or panel is None:
    st.stop()

if not models:
    st.info("Elige al menos un modelo en la barra lateral.", icon="💡")
    st.stop()

of_series = forecasts_all.loc[forecasts_all["unique_id"] == series]
windows_of_series = windows.sort_values("cutoff")

window_labels = {
    int(row.window_id): f"#{row.window_id} · {row.stage} · cutoff {row.cutoff:%Y-%m-%d %Hh}"
    for row in windows_of_series.itertuples()
}
window_choice = st.sidebar.selectbox(
    "Ventana de backtest",
    ["Todas las ventanas", *window_labels.values()],
)
horizon = int(of_series["h_step"].max()) if not of_series.empty else 1
horizon_limit = st.sidebar.slider("Horizonte a mostrar (pasos)", 1, max(horizon, 1), horizon)

selected = of_series.loc[
    of_series["model_id"].isin(models) & (of_series["h_step"] <= horizon_limit)
]
if window_choice != "Todas las ventanas":
    chosen_id = next(wid for wid, label in window_labels.items() if label == window_choice)
    selected = selected.loc[selected["window_id"] == chosen_id]

if selected.empty:
    st.warning(
        "No hay predicciones para esta combinacion de serie, modelos y ventana "
        "(alguna combinacion pudo fallar en el backtest demo, ver A6 en "
        "docs/ARCHITECTURE.md).",
        icon="⚠️",
    )
    st.stop()

context_end = pd.Timestamp(selected["ds"].min())
context_start = context_end - pd.Timedelta(hours=7 * 24)
context = panel.loc[
    (panel["unique_id"] == series) & (panel["ds"] >= context_start) & (panel["ds"] < context_end),
    ["ds", "y"],
]

st.subheader(f"{series}: prediccion frente a historico")
fig = plot_forecast_overlay(
    context, selected, model_colors=palette.model_colors(), unique_id=series
)
st.plotly_chart(widgets.with_rangeslider(fig), width="stretch")

st.subheader("Residuos")
st.plotly_chart(plot_residuals(selected, model_colors=palette.model_colors()), width="stretch")
st.caption(
    "Los siete modelos de esta pagina son el subconjunto que tiene predicciones crudas "
    "persistidas en modo demo (`forecasts_demo.parquet`); el leaderboard completo cubre "
    "mas modelos pero solo con metricas ya agregadas — ver la pagina Leaderboard."
)
