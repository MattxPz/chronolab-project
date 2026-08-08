"""Paletas de color estables por identidad, compartidas entre paginas.

El color de una serie o de un modelo depende de quien es, nunca de la
posicion que ocupa tras un filtro (docs/ARCHITECTURE.md, requisito de la
app: "un modelo = un color en todos los graficos"). Por eso el orden
canonico se calcula una vez sobre el universo completo -todas las series del
panel, todos los modelos del leaderboard, todos los detectores con
scores- y no sobre la seleccion vigente del multiselector.
"""

from __future__ import annotations

import streamlit as st

from chronolab.app.components import state
from chronolab.viz.plots import model_color_map, series_color_map

__all__ = ["detector_colors", "model_colors", "series_colors"]


@st.cache_data(show_spinner=False)
def series_colors() -> dict[str, str]:
    """Color fijo por serie, sobre el universo completo de `state.series_options`."""
    return series_color_map(state.series_options())


@st.cache_data(show_spinner=False)
def model_colors() -> dict[str, str]:
    """Color fijo por modelo, sobre el leaderboard completo (19 modelos).

    Se usa `leaderboard_model_options` y no `model_options` (los siete del
    backtest demo) a proposito: los siete son un subconjunto del leaderboard
    con los mismos `model_id`, y anclar el orden al universo mas grande es lo
    que garantiza que, por ejemplo, ``mstl`` tenga el mismo color en la
    pagina Leaderboard (19 modelos) y en Forecast (7 modelos).
    """
    return model_color_map(state.leaderboard_model_options())


@st.cache_data(show_spinner=False)
def detector_colors() -> dict[str, str]:
    """Color fijo por detector, sobre el universo completo de `state.detector_options`."""
    return series_color_map(state.detector_options())
