"""Leaderboard: metricas, precision frente a coste, Diebold-Mariano y MCS."""

from __future__ import annotations

import streamlit as st

from chronolab.app.components import palette, state, theme, widgets
from chronolab.viz.plots import plot_accuracy_vs_cost, plot_dm_heatmap

st.set_page_config(page_title="Chronolab · Leaderboard", page_icon="⏱️", layout="wide")
theme.apply_density(page_title="leaderboard")

st.title("Leaderboard")
series = widgets.series_selector()

leaderboard = widgets.safe_load(state.load_leaderboard, name="el leaderboard")
if leaderboard is None:
    st.stop()

view = st.radio(
    "Vista",
    ["Agregado (todas las series)", "Por serie seleccionada"],
    horizontal=True,
    key="chronolab_leaderboard_view",
)
por_serie = view == "Por serie seleccionada" and series is not None
if por_serie:
    st.caption(f"Serie: **{series}**")
scope = (
    leaderboard.loc[leaderboard["unique_id"] == series]
    if por_serie
    else leaderboard.loc[leaderboard["unique_id"].isna()]
)
scope = scope.sort_values("mase")

naive_row = scope.loc[scope["model_id"] == "naive"]
naive_mase = float(naive_row["mase"].iloc[0]) if not naive_row.empty else None
top = scope.dropna(subset=["mase"]).iloc[0] if scope["mase"].notna().any() else None

st.markdown("##### En resumen")
cards = st.columns(4)
with cards[0], st.container(border=True):
    st.metric("Modelos evaluados", int(scope["model_id"].nunique()))
with cards[1], st.container(border=True):
    if top is not None:
        delta = None if naive_mase is None else float(top["mase"]) - naive_mase
        st.metric(
            f"Mejor MASE — {top['model_id']}",
            f"{top['mase']:.3f}",
            delta=None if delta is None else f"{delta:+.3f} vs naive",
            delta_color="inverse",
            help=theme.METRIC_HELP["mase"],
        )
    else:
        st.metric("Mejor MASE", "—")
with cards[2], st.container(border=True):
    st.metric("Ventanas del run", int(scope["n_windows"].max()) if not scope.empty else "—")
with cards[3], st.container(border=True):
    zero_shot = int(scope["is_zero_shot"].sum()) if "is_zero_shot" in scope.columns else 0
    st.metric(
        "Zero-shot", zero_shot, help="Modelos sin ajuste local (p. ej. Chronos): `fit_seconds ≈ 0`."
    )

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
column_config = {
    "model_id": st.column_config.TextColumn("Modelo"),
    "mae": st.column_config.NumberColumn("MAE", help=theme.METRIC_HELP["mae"], format="%.4f"),
    "rmse": st.column_config.NumberColumn("RMSE", help=theme.METRIC_HELP["rmse"], format="%.4f"),
    "mape": st.column_config.NumberColumn("MAPE", help=theme.METRIC_HELP["mape"], format="%.3f"),
    "smape": st.column_config.NumberColumn("sMAPE", help=theme.METRIC_HELP["smape"], format="%.3f"),
    "mase": st.column_config.NumberColumn("MASE", help=theme.METRIC_HELP["mase"], format="%.3f"),
    "coverage_95": st.column_config.NumberColumn(
        "Cobertura 95%", help=theme.METRIC_HELP["coverage_95"], format="percent"
    ),
    "coverage_80": st.column_config.NumberColumn(
        "Cobertura 80%", help=theme.METRIC_HELP["coverage_80"], format="percent"
    ),
    "coverage_50": st.column_config.NumberColumn(
        "Cobertura 50%", help=theme.METRIC_HELP["coverage_50"], format="percent"
    ),
    "fit_seconds_total": st.column_config.NumberColumn(
        "Coste entrenamiento (s)", help=theme.METRIC_HELP["fit_seconds_total"], format="%.2f"
    ),
    "is_zero_shot": st.column_config.CheckboxColumn("Zero-shot"),
}
st.dataframe(
    scope[[c for c in display_columns if c in scope.columns]],
    width="stretch",
    hide_index=True,
    column_config=column_config,
)
st.caption(
    "Encabezados ordenables · pasa el cursor sobre un encabezado para ver que significa la metrica."
)

with st.expander("Precision frente a coste computacional", expanded=True):
    cost_scope = leaderboard.loc[leaderboard["unique_id"].isna()].dropna(subset=["mase"])
    st.plotly_chart(
        plot_accuracy_vs_cost(cost_scope, model_colors=palette.model_colors()),
        width="stretch",
    )
    st.caption(
        "Eje y: MASE (mas bajo, mejor). Eje x: coste total de entrenamiento en segundos, "
        "escala log. Agregado sobre todas las series."
    )

with st.expander("Significancia de Diebold-Mariano", expanded=False):
    dm_matrix = widgets.safe_load(state.load_dm_matrix, name="la matriz de Diebold-Mariano")
    if dm_matrix is not None:
        st.plotly_chart(plot_dm_heatmap(dm_matrix), width="stretch")
        st.caption(theme.METRIC_HELP["dm_stat"])
        st.caption(
            "Calculado sobre los siete modelos con predicciones crudas del backtest demo, "
            "error absoluto pooled entre las tres series — una aproximacion, no la "
            "comparacion rigurosa que exige una unica serie por contraste."
        )

theme.footer()
