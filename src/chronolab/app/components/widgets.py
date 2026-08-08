"""Widgets reutilizables: selectores persistentes, slider de alfa y avisos.

Los selectores usan el patron recomendado de Streamlit para estado
persistente: se siembra ``st.session_state[key]`` **antes** de crear el
widget (nunca a la vez que se pasa ``default``/``value``, que Streamlit
rechaza si `key` ya existe) y el widget se crea solo con ``key=``. Como la
clave es la misma cadena en todas las paginas y `st.session_state` es unico
por sesion de navegador, el valor sobrevive al cambio de pagina sin que este
modulo lo copie a mano -es justo lo que pide la especificacion de la app
("la serie seleccionada persiste al cambiar de pagina").
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chronolab.app.components import state
from chronolab.errors import ChronolabError

__all__ = [
    "ALPHA_KEY",
    "DETECTOR_KEY",
    "MODELS_KEY",
    "SERIES_KEY",
    "alpha_slider",
    "detector_selector",
    "missing_artifact",
    "model_multiselect",
    "safe_load",
    "series_selector",
    "with_rangeslider",
]

SERIES_KEY = "chronolab_series"
MODELS_KEY = "chronolab_models"
DETECTOR_KEY = "chronolab_detector"
ALPHA_KEY = "chronolab_alpha"


def series_selector() -> str | None:
    """Selector de serie en la barra lateral, compartido por todas las paginas.

    Returns
    -------
    str or None
        Serie elegida, o ``None`` si `panel.parquet` no esta disponible.
    """
    options = state.series_options()
    if not options:
        return None
    if st.session_state.get(SERIES_KEY) not in options:
        st.session_state[SERIES_KEY] = options[0]
    return str(st.sidebar.selectbox("Serie", options, key=SERIES_KEY))


def model_multiselect(*, label: str = "Modelos", default_n: int = 3) -> list[str]:
    """Multiselector de modelos, restringido a los que tienen predicciones crudas.

    Parameters
    ----------
    label
        Etiqueta del widget.
    default_n
        Cuantos modelos preseleccionar la primera vez que se dibuja.

    Returns
    -------
    list[str]
        Modelos elegidos, en el orden en que los devuelve Streamlit.
    """
    options = state.model_options()
    if not options:
        return []
    current = st.session_state.get(MODELS_KEY)
    pruned = [m for m in current if m in options] if current else []
    st.session_state[MODELS_KEY] = pruned or options[: min(default_n, len(options))]
    return list(st.sidebar.multiselect(label, options, key=MODELS_KEY))


def detector_selector() -> str | None:
    """Selector de detector de anomalias.

    Returns
    -------
    str or None
        Detector elegido, o ``None`` si no hay scores disponibles.
    """
    options = state.detector_options()
    if not options:
        return None
    if st.session_state.get(DETECTOR_KEY) not in options:
        st.session_state[DETECTOR_KEY] = options[0]
    return str(st.sidebar.selectbox("Detector", options, key=DETECTOR_KEY))


def alpha_slider(*, default: float = 0.05) -> float:
    """Slider de nivel alfa: recalcula el umbral sobre scores ya calculados (A5).

    Parameters
    ----------
    default
        Valor inicial, solo se usa la primera vez que se dibuja el widget.

    Returns
    -------
    float
        Nivel alfa vigente, en ``(0, 1)``.
    """
    if ALPHA_KEY not in st.session_state:
        st.session_state[ALPHA_KEY] = default
    return float(
        st.sidebar.slider(
            "Nivel alfa (umbral de deteccion)",
            min_value=0.001,
            max_value=0.20,
            step=0.001,
            format="%.3f",
            key=ALPHA_KEY,
            help="Un score se marca como anomalia si score >= -log10(alfa). "
            "No vuelve a puntuar nada: solo reevalua el umbral sobre los "
            "scores ya calculados por el detector.",
        )
    )


def missing_artifact(name: str, *, detail: str | None = None) -> None:
    """Aviso uniforme para un artefacto ausente: mensaje util, no una traza.

    Parameters
    ----------
    name
        Nombre legible de lo que falta (p. ej. "predicciones del backtest").
    detail
        Texto adicional, tipicamente el mensaje de `ArtifactNotFound`.
    """
    message = f"**{name}** no esta disponible en este modo demo."
    if detail:
        message += f"\n\n{detail}"
    st.info(message, icon="💡")


def safe_load(loader: Callable[[], pd.DataFrame], *, name: str) -> pd.DataFrame | None:
    """Ejecuta un `load_*` de `chronolab.app.components.state` sin propagar la excepcion.

    Parameters
    ----------
    loader
        Funcion sin argumentos que devuelve el DataFrame (p. ej.
        `state.load_forecasts`).
    name
        Nombre legible para el aviso si falla.

    Returns
    -------
    pandas.DataFrame or None
        La tabla, o ``None`` tras haber dibujado `missing_artifact`.
    """
    try:
        return loader()
    except ChronolabError as exc:
        missing_artifact(name, detail=str(exc))
        return None


def with_rangeslider(fig: go.Figure) -> go.Figure:
    """Anade el rangeslider de Plotly al eje x, para las series largas.

    Parameters
    ----------
    fig
        Figura a modificar **en el sitio** (y tambien se devuelve, para
        encadenar).

    Returns
    -------
    plotly.graph_objects.Figure
        La misma `fig`.
    """
    fig.update_xaxes(rangeslider={"visible": True})
    return fig
