"""Overview: serie, descomposicion, estadisticos de dificultad y calidad de datos."""

from __future__ import annotations

import streamlit as st

from chronolab.app.components import palette, state, theme, widgets
from chronolab.viz.plots import (
    plot_difficulty_table,
    plot_mstl,
    plot_quality_overview,
    plot_series_with_flags,
)

st.set_page_config(page_title="Chronolab · Overview", page_icon="⏱️", layout="wide")
theme.apply_density(page_title="overview")

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
st.caption("Arrastra el minigrafico inferior o el rango del eje x para hacer zoom.")

quality = widgets.safe_load(state.load_quality_report, name="el informe de calidad")
difficulty = widgets.safe_load(state.load_difficulty, name="los estadisticos de dificultad")

st.markdown("##### En resumen")
cards = st.columns(6)
if quality is not None:
    row = quality.loc[quality["unique_id"] == series]
    if not row.empty:
        r = row.iloc[0]
        metrics = [
            ("Cobertura", f"{r['coverage']:.1%}", theme.METRIC_HELP["coverage"]),
            ("Huecos", int(r["n_gaps"]), "Filas con `y = NaN` en la rejilla completa de la serie."),
            (
                "Ceros",
                int(r["n_zeros"]),
                "Observaciones exactamente en cero: posible sensor apagado.",
            ),
            (
                "Atipicos",
                int(r["n_outliers"]),
                "Filas con z-score robusto (mediana/MAD) por encima de 4.",
            ),
        ]
        for column, (label, value, help_text) in zip(cards[:4], metrics, strict=True):
            with column, st.container(border=True):
                st.metric(label, value, help=help_text)
if difficulty is not None:
    row = difficulty.loc[difficulty["unique_id"] == series]
    if not row.empty:
        r = row.iloc[0]
        for column, (label, key) in zip(
            cards[4:],
            [("Fuerza tendencia", "trend_strength"), ("Entropia espectral", "spectral_entropy")],
            strict=True,
        ):
            with column, st.container(border=True):
                st.metric(label, f"{r[key]:.2f}", help=theme.METRIC_HELP[key])

with st.expander("Calidad de datos — todas las series", expanded=False):
    if quality is not None:
        st.plotly_chart(plot_quality_overview(quality), width="stretch")
    else:
        widgets.missing_artifact("el informe de calidad")

with st.expander("Dificultad — comparativa entre series", expanded=False):
    if difficulty is not None:
        st.plotly_chart(plot_difficulty_table(difficulty), width="stretch")
        st.caption(
            "Fuerza de tendencia y estacional cerca de 1: la componente domina la serie "
            "(Hyndman & Wang, 2015). Entropia espectral cerca de 1: espectro plano, dificil "
            "de predecir."
        )
    else:
        widgets.missing_artifact("los estadisticos de dificultad")

with st.expander("Descomposicion MSTL", expanded=True):
    mstl = widgets.safe_load(state.load_mstl_components, name="la descomposicion MSTL")
    if mstl is not None:
        series_mstl = mstl.loc[mstl["unique_id"] == series].set_index("ds")
        st.plotly_chart(plot_mstl(series_mstl, periods=(24, 168)), width="stretch")

theme.footer()
