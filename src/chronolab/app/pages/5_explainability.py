"""Explicabilidad: importancia de features, valores SHAP, pesos de atencion del TFT, descomposicion de la prediccion."""

from __future__ import annotations

import streamlit as st

from chronolab.app.components import state, widgets
from chronolab.config import figures_dir
from chronolab.viz.plots import (
    plot_prediction_decomposition,
    plot_tft_temporal_attention,
    plot_tft_variable_attention,
)

st.set_page_config(page_title="Chronolab · Explicabilidad", page_icon="⏱️", layout="wide")

st.title("Explicabilidad")
series = widgets.series_selector()

col_native, col_shap = st.columns(2)
for column, filename, title in (
    (col_native, "07_feature_importance_native.png", "Importancia de features (nativa)"),
    (col_shap, "07_feature_importance_shap.png", "Valores SHAP"),
):
    with column:
        st.subheader(title)
        path = figures_dir() / filename
        if path.exists():
            st.image(str(path), width="stretch")
            st.caption(
                "Figura estatica precomputada (`scripts/run_ml_feature_analysis.py`): la app "
                "no recalcula SHAP en caliente."
            )
        else:
            widgets.missing_artifact(title, detail=f"Falta `reports/figures/{filename}`.")

st.subheader("Atencion del TFT")
tft = widgets.safe_load(state.load_tft_interpretability, name="la interpretabilidad del TFT")
if tft is not None:
    col_variable, col_temporal = st.columns(2)
    with col_variable:
        st.plotly_chart(plot_tft_variable_attention(tft), width="stretch")
    with col_temporal:
        st.plotly_chart(plot_tft_temporal_attention(tft), width="stretch")
    st.caption(
        "Pesos de atencion de un Temporal Fusion Transformer entrenado sobre el dataset "
        "completo (`scripts/run_deep_analysis.py`), no sobre el panel demo pequeno de esta app."
    )

st.subheader("Descomposicion de la prediccion")
mstl = widgets.safe_load(state.load_mstl_components, name="la descomposicion MSTL")
if mstl is not None and series is not None:
    series_mstl = mstl.loc[mstl["unique_id"] == series].sort_values("ds")
    if series_mstl.empty:
        st.info(f"Sin descomposicion para '{series}'.", icon="💡")
    else:
        options = series_mstl["ds"].tail(48).tolist()
        ds = st.select_slider(
            "Instante a descomponer",
            options=options,
            value=options[-1],
            format_func=lambda ts: f"{ts:%Y-%m-%d %Hh}",
        )
        st.plotly_chart(
            plot_prediction_decomposition(series_mstl, unique_id=series, ds=ds),
            width="stretch",
        )
        st.caption(
            "Tendencia + estacional (24h) + estacional (168h) + residuo = valor observado "
            "(descomposicion MSTL precalculada, no la salida interna de un modelo concreto)."
        )
