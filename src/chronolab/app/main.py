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

from chronolab.app.components import state, widgets
from chronolab.config import get_settings

st.set_page_config(page_title="Chronolab", page_icon="⏱️", layout="wide")


def _artifact_status() -> None:
    """Resumen de que artefactos existen en `reports/results/`, sin leerlos."""
    status = state.available()
    labels = {
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
    columns = st.columns(4)
    for i, (name, ok) in enumerate(status.items()):
        with columns[i % 4]:
            st.markdown(f"{'🟢' if ok else '⚪'} {labels.get(name, name)}")


def main() -> None:
    """Dibuja la pagina de inicio."""
    settings = get_settings()

    st.title("⏱️ Chronolab")
    st.caption("Forecasting y deteccion de anomalias en series temporales multivariadas.")

    if settings.demo_mode:
        st.info(
            "**Modo demo.** Los artefactos de esta app son el subconjunto pequeno "
            "versionado en el repositorio (`reports/results/`), no un run completo. "
            "La app nunca entrena ni recalcula: solo lee estos artefactos y dibuja "
            "(docs/ARCHITECTURE.md, A5).",
            icon="🧪",
        )

    widgets.series_selector()
    series = st.session_state.get(widgets.SERIES_KEY)
    if series:
        st.sidebar.caption(f"Serie activa: **{series}** (persiste al cambiar de pagina)")

    st.subheader("Paginas")
    pages = [
        ("1  Overview", "Serie, descomposicion MSTL, calidad de datos y dificultad."),
        ("2  Forecast", "Predicciones superpuestas con bandas de incertidumbre y residuos."),
        ("3  Leaderboard", "Metricas, precision frente a coste y Diebold-Mariano."),
        ("4  Anomalias", "Serie marcada, umbral ajustable y tabla de eventos."),
        ("5  Explicabilidad", "Importancia de features, atencion del TFT y descomposicion."),
    ]
    for name, description in pages:
        st.markdown(f"**{name}** — {description}")

    st.divider()
    st.subheader("Artefactos disponibles")
    _artifact_status()
    st.caption(
        "🟢 disponible · ⚪ ausente. Los artefactos demo se generan con "
        "`uv run --extra ml python scripts/build_demo_artifacts.py`; los de "
        "anomalias, con `scripts/run_anomaly_eval.py`."
    )


main()
