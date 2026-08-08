"""El arbol de modulos existe y coincide con docs/ARCHITECTURE.md §2.

No es un test de relleno: el documento de arquitectura es la referencia de todos
los prompts siguientes, y un modulo que se renombra sin actualizarlo convierte el
documento en ficcion.
"""

from __future__ import annotations

import ast
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
    "chronolab.evaluation.tuning",
    "chronolab.artifacts",
    "chronolab.artifacts.reader",
    "chronolab.artifacts.schemas",
    "chronolab.artifacts.writer",
    "chronolab.viz",
    "chronolab.viz.plots",
    "chronolab.app",
    "chronolab.app.components",
    "chronolab.api",
    "chronolab.api.service",
]

# Los scripts de Streamlit se comprueban como ficheros, no importandolos, por
# dos razones distintas:
#
# - Las paginas de `app/pages/` se nombran por convencion del framework
#   (`1_overview.py`) y no son identificadores de modulo validos.
# - `app/main.py` si lo es, pero importarlo **ejecuta la app entera**: en un
#   entrypoint de Streamlit `st.set_page_config` y el cuerpo de la pagina corren
#   al importar, que es como el framework espera que se escriba. Ademas
#   `streamlit` vive en el extra `app`, que el job de CI de lint/typecheck/test
#   no instala (ver `.github/workflows/ci.yml`).
#
# Parsear el fichero con `ast` comprueba exactamente lo mismo que el test de
# modulos -que existe y que documenta su proposito- sin importar ni ejecutar
# nada. Es deliberadamente distinto de un `pytest.importorskip("streamlit")`,
# que es el patron del resto de la suite para los extras: aqui saltarse la
# comprobacion dejaria el arbol de `app/` sin verificar precisamente en el
# entorno donde CI la corre.
STREAMLIT_SCRIPTS = [
    "app/main.py",
    "app/pages/1_overview.py",
    "app/pages/2_forecast.py",
    "app/pages/3_leaderboard.py",
    "app/pages/4_anomalies.py",
    "app/pages/5_explainability.py",
]

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "chronolab"


@pytest.mark.parametrize("name", MODULES)
def test_el_modulo_existe_y_documenta_su_proposito(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} no tiene docstring de proposito"


@pytest.mark.parametrize("relative_path", STREAMLIT_SCRIPTS)
def test_el_script_de_streamlit_existe_y_documenta_su_proposito(relative_path: str) -> None:
    path = PACKAGE_ROOT / relative_path
    assert path.is_file(), f"falta {relative_path}"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{relative_path} no tiene docstring de proposito"


def test_el_paquete_distribuye_tipos() -> None:
    assert (PACKAGE_ROOT / "py.typed").is_file()
