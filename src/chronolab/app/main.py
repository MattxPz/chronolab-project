"""Entrada de Streamlit: navegacion, seleccion de run y dataset, estado de sesion.

Solo lee artefactos de ``reports/results/`` (docs/ARCHITECTURE.md A5: la app
no entrena ni recalcula) y solo importa `chronolab.artifacts.reader`,
`chronolab.viz`, `chronolab.config` y `chronolab.types`, indirectamente a
traves de `chronolab.app.components` -nunca `chronolab.models`,
`chronolab.evaluation` ni `chronolab.anomaly`. Esa restriccion es tambien la
que mantiene el arranque bajo tres segundos: nada de lo que importa esta
pagina arrastra `torch`, `prophet` ni `statsforecast`.

Streamlit descubre las paginas de ``app/pages/`` automaticamente; este
archivo es la pagina raiz (el "Home") y el sitio donde se fija por primera
vez la serie compartida, para que quien entre directo a "Forecast" sin pasar
por aqui la vea ya inicializada.
"""

from __future__ import annotations

import streamlit as st

from chronolab.app.components import state, theme, widgets
from chronolab.config import get_settings

st.set_page_config(page_title="Chronolab", page_icon="⏱️", layout="wide")
theme.apply_density(page_title="home")

_PAGES = [
    (
        "1_overview.py",
        "🔎",
        "Overview",
        "Serie, descomposicion MSTL, calidad de datos y dificultad.",
    ),
    (
        "2_forecast.py",
        "📈",
        "Forecast",
        "Predicciones superpuestas con bandas de incertidumbre y residuos.",
    ),
    (
        "3_leaderboard.py",
        "🏆",
        "Leaderboard",
        "Metricas, precision frente a coste y Diebold-Mariano.",
    ),
    ("4_anomalies.py", "🚨", "Anomalias", "Serie marcada, umbral ajustable y tabla de eventos."),
    (
        "5_explainability.py",
        "🧭",
        "Explicabilidad",
        "Importancia de features, atencion del TFT y descomposicion.",
    ),
]

_ARTIFACT_LABELS: dict[str, str] = {
    "leaderboard": "Leaderboard",
    "anomaly_scores": "Scores de anomalias",
    "anomaly_results": "Metricas de deteccion",
    "anomaly_truth": "Eventos anomalos (ground truth)",
    "tft_interpretability": "Interpretabilidad TFT",
    "panel": "Serie demo",
    "quality_report": "Informe de calidad",
    "quality_outliers": "Atipicos",
    "mstl_components": "Descomposicion MSTL",
    "difficulty": "Dificultad de la serie",
    "forecasts": "Predicciones del backtest demo",
    "windows": "Ventanas del backtest demo",
    "dm_matrix": "Diebold-Mariano",
}


def _hero() -> None:
    """Titulo, subtitulo y el aviso de modo demo."""
    settings = get_settings()
    st.title("⏱️ Chronolab")
    st.caption(
        "Forecasting y deteccion de anomalias en series temporales multivariadas — "
        "panel de control de resultados, no un cuaderno de analisis."
    )
    if settings.demo_mode:
        st.info(
            "**Modo demo.** Los artefactos de esta app son el subconjunto pequeno "
            "versionado en el repositorio (`reports/results/`), no un run completo. "
            "La app nunca entrena ni recalcula: solo lee estos artefactos y dibuja "
            "(docs/ARCHITECTURE.md, A5).",
            icon="🧪",
        )


def _headline_metrics() -> None:
    """Cuatro cifras clave, como tarjetas, para orientar antes de navegar."""
    series = state.series_options()
    leaderboard_models = state.leaderboard_model_options()
    forecast_models = state.model_options()
    detectors = state.detector_options()

    columns = st.columns(4)
    values = [
        ("Series demo", len(series), "s00 · s01 · s02"),
        ("Modelos en el leaderboard", len(leaderboard_models), "metricas agregadas"),
        ("Modelos con predicciones crudas", len(forecast_models), "backtest demo"),
        (
            "Detectores de anomalias",
            len(detectors),
            "conformal, isoforest, LSTM-AE, matrix profile",
        ),
    ]
    for column, (label, value, help_text) in zip(columns, values, strict=True):
        with column, st.container(border=True):
            st.metric(label, value if value else "—", help=help_text)


def _page_cards() -> None:
    """Una tarjeta por pagina, con enlace de navegacion nativo."""
    columns = st.columns(len(_PAGES))
    for column, (target, icon, name, description) in zip(columns, _PAGES, strict=True):
        with column, st.container(border=True):
            st.markdown(f"### {icon} {name}")
            st.caption(description)
            st.page_link(f"pages/{target}", label="Abrir", icon="➡️")


def _artifact_status() -> None:
    """Detalle de que artefactos existen en `reports/results/`, en un expander."""
    status = state.available()
    with st.expander(f"Artefactos disponibles ({sum(status.values())}/{len(status)})"):
        columns = st.columns(3)
        for i, (name, ok) in enumerate(status.items()):
            with columns[i % 3]:
                st.markdown(f"{'🟢' if ok else '⚪'} {_ARTIFACT_LABELS.get(name, name)}")
        st.caption(
            "🟢 disponible · ⚪ ausente. Los artefactos demo se generan con "
            "`uv run --extra ml python scripts/build_demo_artifacts.py`; los de "
            "anomalias, con `scripts/run_anomaly_eval.py`."
        )


def main() -> None:
    """Dibuja la pagina de inicio."""
    _hero()

    widgets.series_selector()
    series = st.session_state.get(widgets.SERIES_KEY)
    if series:
        st.sidebar.caption(f"Serie activa: **{series}** (persiste al cambiar de pagina)")

    _headline_metrics()
    st.subheader("Paginas")
    _page_cards()
    _artifact_status()
    theme.footer()


main()
