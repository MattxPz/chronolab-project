"""Forecast: predicciones superpuestas con bandas, por serie, modelo y horizonte."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chronolab.app.components import palette, state, theme, widgets
from chronolab.viz.plots import plot_forecast_overlay, plot_residuals

st.set_page_config(page_title="Chronolab · Forecast", page_icon="⏱️", layout="wide")
theme.apply_density(page_title="forecast")

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
window_options = ["Todas las ventanas", *window_labels.values()]
window_key = "chronolab_forecast_window"
if st.session_state.get(window_key) not in window_options:
    st.session_state[window_key] = window_options[0]
window_choice = st.sidebar.selectbox("Ventana de backtest", window_options, key=window_key)
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
st.caption("Arrastra el minigrafico inferior para hacer zoom en un tramo del horizonte.")

leaderboard = widgets.safe_load(state.load_leaderboard, name="el leaderboard")
if leaderboard is not None:
    baseline_row = leaderboard.loc[
        (leaderboard["model_id"] == "naive") & (leaderboard["unique_id"] == series)
    ]
    baseline_mase = float(baseline_row["mase"].iloc[0]) if not baseline_row.empty else None

    st.markdown("##### MASE en esta serie (holdout), frente al naive")
    cards = st.columns(min(len(models), 6))
    shown = sorted(models)[: len(cards)]
    for column, model_id in zip(cards, shown, strict=True):
        row = leaderboard.loc[
            (leaderboard["model_id"] == model_id) & (leaderboard["unique_id"] == series)
        ]
        with column, st.container(border=True):
            if row.empty:
                st.metric(model_id, "—", help="Sin fila de leaderboard para esta serie.")
                continue
            mase = float(row["mase"].iloc[0])
            delta = None if baseline_mase is None else mase - baseline_mase
            st.metric(
                model_id,
                f"{mase:.3f}",
                delta=None if delta is None else f"{delta:+.3f} vs naive",
                delta_color="inverse",  # MASE mas bajo es mejor: una delta negativa es una mejora
                help=theme.METRIC_HELP["mase"],
            )
    if len(models) > len(cards):
        st.caption(
            f"+{len(models) - len(cards)} modelo(s) mas seleccionados, sin tarjeta por espacio."
        )

with st.expander("Residuos (real menos prediccion)", expanded=True):
    st.plotly_chart(plot_residuals(selected, model_colors=palette.model_colors()), width="stretch")
    st.caption(
        "Puntos cerca de la linea cero son mejores. Un patron sistematico (no ruido) indica sesgo del modelo."
    )

st.caption(
    "Los siete modelos de esta pagina son el subconjunto que tiene predicciones crudas "
    "persistidas en modo demo (`forecasts_demo.parquet`); el leaderboard completo cubre "
    "mas modelos pero solo con metricas ya agregadas — ver la pagina Leaderboard."
)
theme.footer()
