"""El arbol de modulos existe y coincide con docs/ARCHITECTURE.md §2.

No es un test de relleno: el documento de arquitectura es la referencia de todos
los prompts siguientes, y un modulo que se renombra sin actualizarlo convierte el
documento en ficcion.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

MODULES = [
    "chronolab",
    "chronolab.config",
    "chronolab.errors",
    "chronolab.logging",
    "chronolab.panel",
    "chronolab.types",
    "chronolab.data",
    "chronolab.data.align",
    "chronolab.data.assemble",
    "chronolab.data.cache",
    "chronolab.data.calendar",
    "chronolab.data.futr",
    "chronolab.data.protocols",
    "chronolab.data.quality",
    "chronolab.data.schemas",
    "chronolab.data.sources",
    "chronolab.data.sources.binance",
    "chronolab.data.sources.open_meteo",
    "chronolab.data.sources.ree",
    "chronolab.data.sources.synthetic",
    "chronolab.data.sources.uci_electricity",
    "chronolab.features",
    "chronolab.features.builders",
    "chronolab.features.ops",
    "chronolab.features.roles",
    "chronolab.models",
    "chronolab.models.baselines",
    "chronolab.models.protocols",
    "chronolab.models.registry",
    "chronolab.models.wrappers",
    "chronolab.models.adapters",
    "chronolab.models.adapters.chronos",
    "chronolab.models.adapters.mlforecast",
    "chronolab.models.adapters.neuralforecast",
    "chronolab.models.adapters.prophet",
    "chronolab.models.adapters.statsforecast",
    "chronolab.models.adapters.torch_lstm",
    "chronolab.models.torch",
    "chronolab.models.torch.dataset",
    "chronolab.models.torch.modules",
    "chronolab.models.torch.trainer",
    "chronolab.anomaly",
    "chronolab.anomaly.autoencoder",
    "chronolab.anomaly.conformal",
    "chronolab.anomaly.events",
    "chronolab.anomaly.injection",
    "chronolab.anomaly.isolation",
    "chronolab.anomaly.matrix_profile",
    "chronolab.anomaly.protocols",
    "chronolab.anomaly.thresholds",
    "chronolab.evaluation",
    "chronolab.evaluation.aggregate",
    "chronolab.evaluation.anomaly_metrics",
    "chronolab.evaluation.backtest",
    "chronolab.evaluation.metrics",
    "chronolab.evaluation.splitters",
    "chronolab.evaluation.stats_tests",
    "chronolab.artifacts",
    "chronolab.artifacts.reader",
    "chronolab.artifacts.schemas",
    "chronolab.artifacts.writer",
    "chronolab.viz",
    "chronolab.viz.plots",
    "chronolab.app",
    "chronolab.app.main",
    "chronolab.app.components",
    "chronolab.api",
    "chronolab.api.service",
]

# Las paginas de Streamlit se nombran por convencion del framework y no son
# identificadores validos, asi que se comprueban como ficheros y no como modulos.
STREAMLIT_PAGES = [
    "1_overview.py",
    "2_forecast.py",
    "3_leaderboard.py",
    "4_anomalies.py",
    "5_explainability.py",
]

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "chronolab"


@pytest.mark.parametrize("name", MODULES)
def test_el_modulo_existe_y_documenta_su_proposito(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} no tiene docstring de proposito"


@pytest.mark.parametrize("page", STREAMLIT_PAGES)
def test_la_pagina_de_streamlit_existe(page: str) -> None:
    assert (PACKAGE_ROOT / "app" / "pages" / page).is_file()


def test_el_paquete_distribuye_tipos() -> None:
    assert (PACKAGE_ROOT / "py.typed").is_file()
