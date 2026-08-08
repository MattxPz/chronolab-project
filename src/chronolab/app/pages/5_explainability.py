"""Explicabilidad: importancia de features, valores SHAP, pesos de atencion del TFT, descomposicion de la prediccion."""

from __future__ import annotations

import streamlit as st

from chronolab.app.components import state, theme, widgets
from chronolab.config import figures_dir
from chronolab.viz.plots import (
    plot_prediction_decomposition,
    plot_tft_temporal_attention,
    plot_tft_variable_attention,
)

st.set_page_config(page_title="Chronolab · Explicabilidad", page_icon="⏱️", layout="wide")
theme.apply_density(page_title="explainability")

st.title("Explicabilidad")
series = widgets.series_selector()

st.subheader("Atencion del TFT")
tft = widgets.safe_load(state.load_tft_interpretability, name="la interpretabilidad del TFT")
if tft is not None:
    col_variable, col_temporal = st.columns(2)
    with col_variable, st.container(border=True):
        st.markdown("**Importancia de variables**")
        st.plotly_chart(plot_tft_variable_attention(tft), width="stretch")
        st.caption(
            "Cuanto peso reparte el modelo entre `temp_c` y su propio historial, por bloque pasado/futuro."
        )
    with col_temporal, st.container(border=True):
        st.markdown("**Atencion temporal**")
        st.plotly_chart(plot_tft_temporal_attention(tft), width="stretch")
        st.caption("Que instantes del contexto y del horizonte pesan mas en la prediccion final.")
    st.caption(
        "Pesos de atencion de un Temporal Fusion Transformer entrenado sobre el dataset "
        "completo (`scripts/run_deep_analysis.py`), no sobre el panel demo pequeno de esta app."
    )
else:
    widgets.missing_artifact("la interpretabilidad del TFT")

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

with st.expander("Importancia de features y SHAP (referencia estatica)", expanded=False):
    st.caption(
        "Figuras precomputadas sobre el dataset completo (`scripts/run_ml_feature_analysis.py`), "
        "no sobre el panel demo: la app no recalcula SHAP en caliente."
    )
    col_native, col_shap = st.columns(2)
    for column, filename, title in (
        (col_native, "07_feature_importance_native.png", "Importancia nativa (LightGBM)"),
        (col_shap, "07_feature_importance_shap.png", "Valores SHAP"),
    ):
        with column:
            st.markdown(f"**{title}**")
            path = figures_dir() / filename
            if path.exists():
                st.image(str(path), width="stretch")
            else:
                widgets.missing_artifact(title, detail=f"Falta `reports/figures/{filename}`.")

theme.footer()
