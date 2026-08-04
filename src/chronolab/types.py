"""Tipos base del proyecto: identificadores, roles y vintages.

Modulo hoja: no importa nada de `chronolab`. Todo lo demas puede importarlo sin
riesgo de ciclos.
"""

from enum import StrEnum
from typing import Literal, NewType

__all__ = [
    "DatasetId",
    "DetectorId",
    "ModelId",
    "RefitCost",
    "Role",
    "RunId",
    "SeriesId",
    "SplitMode",
    "Stage",
    "Vintage",
]

# Identificadores. Son `NewType` y no `str` para que mypy distinga un ModelId de
# un DetectorId: son cadenas, pero intercambiarlos es un bug.
RunId = NewType("RunId", str)
"""Identificador de run. ULID, ordenable por tiempo de creacion."""

DatasetId = NewType("DatasetId", str)
"""Identificador estable de dataset. Clave de particion de artefactos."""

ModelId = NewType("ModelId", str)
"""Identificador estable de modelo. Clave de particion de `forecasts`."""

DetectorId = NewType("DetectorId", str)
"""Identificador estable de detector. Clave de particion de `anomaly_scores`."""

SeriesId = NewType("SeriesId", str)
"""Identificador de serie dentro de un panel (`unique_id`)."""


class Role(StrEnum):
    """Papel semantico de una columna dentro del panel.

    El formato largo de Nixtla no distingue estos roles, y esa es exactamente su
    carencia critica: la confusion entre `FUTR_EXOG` y `HIST_EXOG` es la fuente
    principal de fuga de informacion temporal.
    """

    TARGET = "target"
    """Columna objetivo. Por convencion Nixtla se llama ``y``."""

    FUTR_EXOG = "futr_exog"
    """Exogena conocida a futuro en el instante de predecir (calendario, prevision)."""

    HIST_EXOG = "hist_exog"
    """Exogena conocida solo hasta el cutoff."""

    STATIC_EXOG = "static_exog"
    """Atributo constante por serie."""


class Vintage(StrEnum):
    """Semantica temporal del valor de una exogena futura.

    Distinguirlos no es un refinamiento: usar `REALIZED` como si fuese una
    prevision infla sistematicamente a los modelos que mas dependen de la
    exogena, y no deja ningun sintoma visible.
    """

    REALIZED = "realized"
    """Valor observado a posteriori. Presciencia perfecta: es una cota superior, no un resultado."""

    ARCHIVED_FORECAST = "archived_forecast"
    """La prevision que realmente existia en el cutoff. El numero honesto."""

    SIMULATED_FORECAST = "simulated_forecast"
    """Realizado degradado con un error sintetico calibrado que crece con el adelanto."""


Stage = Literal["dev", "holdout"]
"""Etapa de una ventana. El tuning solo ve `dev`; el leaderboard solo publica `holdout`."""

SplitMode = Literal["expanding", "sliding"]
"""Modo de la ventana de entrenamiento en el origen rodante."""

RefitCost = Literal["free", "cheap", "expensive"]
"""Coste declarado de reajustar un modelo. Determina la politica de refit por defecto."""
