"""chronolab: forecasting y deteccion de anomalias en series temporales.

El valor del proyecto es el arnes de evaluacion, no los modelos. La arquitectura
completa esta en `docs/ARCHITECTURE.md`, que es la referencia normativa.
"""

from chronolab.errors import ChronolabError
from chronolab.panel import FutrFrame, Panel, PanelSpec
from chronolab.types import (
    DatasetId,
    DetectorId,
    ModelId,
    Role,
    RunId,
    SeriesId,
    Stage,
    Vintage,
)

__version__ = "0.1.0"

__all__ = [
    "ChronolabError",
    "DatasetId",
    "DetectorId",
    "FutrFrame",
    "ModelId",
    "Panel",
    "PanelSpec",
    "Role",
    "RunId",
    "SeriesId",
    "Stage",
    "Vintage",
    "__version__",
]
