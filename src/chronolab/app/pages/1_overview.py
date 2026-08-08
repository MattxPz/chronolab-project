"""Overview: serie, descomposicion, estadisticos de dificultad y calidad de datos."""

from __future__ import annotations

import streamlit as st

from chronolab.app.components import palette, state, widgets
from chronolab.viz.plots import (
    plot_difficulty_table,
    plot_mstl,
    plot_quality_overview,
    plot_series_with_flags,
)

st.set_page_config(page_title="Chronolab · Overview", page_icon="⏱️", layout="wide")

st.title("Overview")
series = widgets.series_selector()

if series is None:
    widgets.missing_artifact(
        "la serie demo",
        detail="Genera `panel.parquet` con `uv run --extra ml python scripts/build_demo_artifacts.py`.",
    )
    st.stop()

panel = widgets.safe_load(state.load_panel, name="la serie demo")
if panel is None:
    st.stop()

series_frame = panel.loc[panel["unique_id"] == series, ["ds", "y"]]
color = palette.series_colors().get(series, "#2a78d6")

st.subheader(f"Serie: {series}")
outliers = state.load_quality_outliers()
fig = plot_series_with_flags(series_frame, outliers, unique_id=series, color=color)
st.plotly_chart(widgets.with_rangeslider(fig), width="stretch")

col_quality, col_difficulty = st.columns(2)

with col_quality:
    st.subheader("Calidad de datos")
    quality = widgets.safe_load(state.load_quality_report, name="el informe de calidad")
    if quality is not None:
        st.plotly_chart(plot_quality_overview(quality), width="stretch")
        row = quality.loc[quality["unique_id"] == series]
        if not row.empty:
            r = row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cobertura", f"{r['coverage']:.1%}")
            m2.metric("Huecos", int(r["n_gaps"]))
            m3.metric("Ceros", int(r["n_zeros"]))
            m4.metric("Atipicos", int(r["n_outliers"]))

with col_difficulty:
    st.subheader("Dificultad de la serie")
    difficulty = widgets.safe_load(state.load_difficulty, name="los estadisticos de dificultad")
    if difficulty is not None:
        st.plotly_chart(plot_difficulty_table(difficulty), width="stretch")
        st.caption(
            "Fuerza de tendencia y estacional cerca de 1: la componente domina la serie "
            "(Hyndman & Wang, 2015). Entropia espectral cerca de 1: espectro plano, dificil "
            "de predecir."
        )

st.subheader("Descomposicion MSTL")
mstl = widgets.safe_load(state.load_mstl_components, name="la descomposicion MSTL")
if mstl is not None:
    series_mstl = mstl.loc[mstl["unique_id"] == series].set_index("ds")
    st.plotly_chart(plot_mstl(series_mstl, periods=(24, 168)), width="stretch")
