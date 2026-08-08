"""Carga cacheada de artefactos y universos de seleccion (series, modelos, detectores).

Streamlit rearranca el script de la pagina en cada interaccion. Sin cache,
cada `st.selectbox` recorrido volveria a leer parquet del disco. Cada
`load_*` de este modulo envuelve exactamente una funcion de
`chronolab.artifacts.reader` con ``st.cache_data``: la app no llama a
``pandas.read_parquet`` ni a `chronolab.artifacts.reader` en ningun otro
sitio (docs/ARCHITECTURE.md §2.1, `app -> artifacts.reader`).

Los ``*_options`` derivan universos estables (para que un color de
`chronolab.app.components.palette` no cambie al filtrar) a partir de esas
tablas cacheadas.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chronolab.artifacts import reader
from chronolab.errors import ChronolabError

__all__ = [
    "available",
    "detector_options",
    "leaderboard_model_options",
    "load_anomaly_results",
    "load_anomaly_scores",
    "load_anomaly_truth",
    "load_difficulty",
    "load_dm_matrix",
    "load_forecasts",
    "load_leaderboard",
    "load_mstl_components",
    "load_panel",
    "load_quality_outliers",
    "load_quality_report",
    "load_tft_interpretability",
    "load_windows",
    "model_options",
    "series_options",
]


@st.cache_data(show_spinner=False)
def available() -> dict[str, bool]:
    """Que artefactos de `chronolab.artifacts.reader.ARTIFACT_FILES` existen."""
    return reader.available_artifacts()


@st.cache_data(show_spinner="Cargando leaderboard...")
def load_leaderboard() -> pd.DataFrame:
    """Metricas por modelo, serie y etapa. `chronolab.artifacts.reader.load_leaderboard`."""
    return reader.load_leaderboard()


@st.cache_data(show_spinner="Cargando scores de anomalias...")
def load_anomaly_scores() -> pd.DataFrame:
    """Scores crudos de los detectores. `chronolab.artifacts.reader.load_anomaly_scores`."""
    return reader.load_anomaly_scores()


@st.cache_data(show_spinner="Cargando metricas de deteccion...")
def load_anomaly_results() -> pd.DataFrame:
    """Metricas de deteccion. `chronolab.artifacts.reader.load_anomaly_results`."""
    return reader.load_anomaly_results()


@st.cache_data(show_spinner="Cargando eventos anomalos...")
def load_anomaly_truth() -> pd.DataFrame:
    """Ground truth de la inyeccion. `chronolab.artifacts.reader.load_anomaly_truth`."""
    return reader.load_anomaly_truth()


@st.cache_data(show_spinner="Cargando interpretabilidad del TFT...")
def load_tft_interpretability() -> pd.DataFrame:
    """Atencion del TFT. `chronolab.artifacts.reader.load_tft_interpretability`."""
    return reader.load_tft_interpretability()


@st.cache_data(show_spinner="Cargando la serie...")
def load_panel() -> pd.DataFrame:
    """Panel demo (con anomalias inyectadas). `chronolab.artifacts.reader.load_panel`."""
    return reader.load_panel()


@st.cache_data(show_spinner="Cargando informe de calidad...")
def load_quality_report() -> pd.DataFrame:
    """Informe de calidad por serie. `chronolab.artifacts.reader.load_quality_report`."""
    return reader.load_quality_report()


@st.cache_data(show_spinner=False)
def load_quality_outliers() -> pd.DataFrame:
    """Filas atipicas. `chronolab.artifacts.reader.load_quality_outliers`."""
    return reader.load_quality_outliers()


@st.cache_data(show_spinner="Cargando la descomposicion MSTL...")
def load_mstl_components() -> pd.DataFrame:
    """Descomposicion MSTL precalculada. `chronolab.artifacts.reader.load_mstl_components`."""
    return reader.load_mstl_components()


@st.cache_data(show_spinner=False)
def load_difficulty() -> pd.DataFrame:
    """Estadisticos de dificultad por serie. `chronolab.artifacts.reader.load_difficulty`."""
    return reader.load_difficulty()


@st.cache_data(show_spinner="Cargando predicciones...")
def load_forecasts() -> pd.DataFrame:
    """Predicciones crudas del backtest demo. `chronolab.artifacts.reader.load_forecasts`."""
    return reader.load_forecasts()


@st.cache_data(show_spinner=False)
def load_windows() -> pd.DataFrame:
    """Ventanas del backtest demo. `chronolab.artifacts.reader.load_windows`."""
    return reader.load_windows()


@st.cache_data(show_spinner=False)
def load_dm_matrix() -> pd.DataFrame:
    """Diebold-Mariano por pareja de modelos. `chronolab.artifacts.reader.load_dm_matrix`."""
    return reader.load_dm_matrix()


# --------------------------------------------------------------------------- #
# Universos de seleccion
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def series_options() -> list[str]:
    """Identificadores de serie del panel demo, ordenados (orden = color)."""
    try:
        panel = load_panel()
    except ChronolabError:
        return []
    return sorted(panel["unique_id"].astype(str).unique())


@st.cache_data(show_spinner=False)
def model_options() -> list[str]:
    """Modelos con predicciones crudas en el backtest demo (Forecast, matriz DM).

    Subconjunto pequeno de `leaderboard_model_options`: solo estos siete
    tienen `forecasts_demo.parquet`. El leaderboard completo (19 modelos) no
    guarda predicciones crudas, solo metricas ya agregadas.
    """
    try:
        forecasts = load_forecasts()
    except ChronolabError:
        return []
    return sorted(forecasts["model_id"].astype(str).unique())


@st.cache_data(show_spinner=False)
def leaderboard_model_options() -> list[str]:
    """Todos los modelos del leaderboard completo, para la pagina Leaderboard."""
    try:
        leaderboard = load_leaderboard()
    except ChronolabError:
        return []
    return sorted(leaderboard["model_id"].astype(str).unique())


@st.cache_data(show_spinner=False)
def detector_options() -> list[str]:
    """Detectores de anomalias con scores disponibles."""
    try:
        scores = load_anomaly_scores()
    except ChronolabError:
        return []
    return sorted(scores["detector_id"].astype(str).unique())
