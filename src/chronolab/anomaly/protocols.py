"""Protocolos `AnomalyDetector`, `FittedDetector` y `Thresholder`.

Puntuar y etiquetar son operaciones distintas y estan separadas a proposito:
VUS-PR y las curvas precision-recall necesitan el score continuo, mientras que
F1 por rangos necesita etiquetas. Si el detector devolviera etiquetas se perderia
irreversiblemente la informacion que exige la metrica principal.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from chronolab.panel import PanelSpec
from chronolab.types import DetectorId, ModelId, RefitCost

__all__ = [
    "AnomalyDetector",
    "DetectorRequirements",
    "FittedDetector",
    "FittedThresholder",
    "ScoringFrame",
    "Thresholder",
]


@dataclass(frozen=True, slots=True)
class ScoringFrame:
    """Tramo de serie con predicciones **fuera de muestra** alineadas.

    Es la unica entrada de los detectores. Lo construye exclusivamente
    ``chronolab.artifacts.reader.scoring_frame`` a partir de la tabla `forecasts`
    de un run, es decir, a partir de predicciones que por construccion son fuera
    de muestra. Un detector no puede recibir predicciones dentro de muestra
    porque no existe ningun camino de codigo que se las entregue.

    Attributes
    ----------
    df
        ``unique_id``, ``ds``, ``y``, ``y_hat``, columnas de cuantil y,
        opcionalmente, las exogenas del panel. Rejilla completa y ordenada.
    spec
        Especificacion del panel del que procede.
    model_id
        Modelo del que provienen ``y_hat`` y los cuantiles. ``None`` para
        detectores que no usan prediccion. Forma parte del identificador efectivo
        del detector en los artefactos, porque "IsolationForest sobre residuos de
        MSTL" y "sobre residuos de NHITS" son detectores distintos.
    start, end
        Extremos **inclusivos** del tramo cubierto.
    """

    df: pd.DataFrame
    spec: PanelSpec
    model_id: ModelId | None
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True, slots=True)
class DetectorRequirements:
    """Necesidades declaradas de un detector.

    Parameters
    ----------
    needs_forecast
        Requiere ``y_hat`` no nulo.
    needs_quantiles
        Requiere columnas de cuantil no nulas.
    window
        Puntos de contexto que consume por score. Determina el calentamiento, es
        decir, cuantos instantes iniciales quedan sin puntuar.
    needs_calibration
        ``False`` para metodos sin ajuste, como Matrix Profile.
    fit_cost
        Coste declarado de calibracion.
    """

    needs_forecast: bool = False
    needs_quantiles: bool = False
    window: int = 1
    needs_calibration: bool = True
    fit_cost: RefitCost = "cheap"


@runtime_checkable
class AnomalyDetector(Protocol):
    """Configuracion de un detector. Inmutable y sin estado calibrado."""

    @property
    def detector_id(self) -> DetectorId:
        """Identificador estable. Clave de particion de `anomaly_scores`."""
        ...

    @property
    def requires(self) -> DetectorRequirements:
        """Necesidades declaradas. Constante."""
        ...

    def fit(self, calib: ScoringFrame) -> "FittedDetector":
        """Calibra el detector con un tramo **anterior** al que se puntuara.

        Parameters
        ----------
        calib
            Tramo de calibracion. Para el detector conformal son los residuos que
            definen el cuantil; para IsolationForest y el autoencoder es el
            conjunto de ajuste; los metodos con ``needs_calibration=False`` solo
            lo usan para fijar su cutoff.

        Returns
        -------
        FittedDetector
            Con ``cutoff = calib.end``.
        """
        ...


@runtime_checkable
class FittedDetector(Protocol):
    """Detector calibrado hasta un instante concreto."""

    @property
    def detector_id(self) -> DetectorId:
        """Identificador del detector del que procede."""
        ...

    @property
    def cutoff(self) -> pd.Timestamp:
        """Ultimo instante usado en calibracion. `score` exige ``ds > cutoff``."""
        ...

    def score(self, frame: ScoringFrame) -> pd.DataFrame:
        """Puntua cada marca de tiempo del tramo.

        Parameters
        ----------
        frame
            Tramo a puntuar. Debe cumplir ``frame.start > cutoff``.

        Returns
        -------
        pandas.DataFrame
            Una fila por ``(unique_id, ds)`` de la entrada, sin excepcion, con
            estas columnas:

            ``score`` : float32
                Grado de anomalia, mayor es mas anomalo. Es una magnitud
                **ordinal dentro de un par (detector, serie)** y no es comparable
                entre detectores ni entre series. No se exige calibrarla a una
                escala comun porque exigirlo seria falso: el error de
                reconstruccion de un autoencoder y un p-valor conformal no viven
                en la misma escala, y forzarlos a ella es justo lo que hace que
                las comparativas de detectores publicadas no signifiquen nada.
            ``scorable`` : bool
                ``False`` en el calentamiento, es decir en los primeros
                ``requires.window - 1`` puntos, o donde ``y`` es ``NaN``. Donde es
                ``False``, ``score`` es ``NaN``.

        Raises
        ------
        CutoffViolation
            Si ``frame.start`` es anterior o igual al cutoff.

        Notes
        -----
        Antes de comparar detectores, `evaluation.anomaly_metrics` interseca las
        mascaras ``scorable`` de todos ellos y los evalua sobre el soporte comun.
        Sin eso, un detector de ventana larga saldria favorecido solo por haberse
        saltado el arranque de la serie.
        """
        ...


@runtime_checkable
class Thresholder(Protocol):
    """Convierte scores continuos en umbrales para una rejilla de alfa."""

    def fit(self, calib_scores: pd.DataFrame) -> "FittedThresholder":
        """Calibra los umbrales con scores de un tramo anterior.

        Parameters
        ----------
        calib_scores
            Columnas ``unique_id``, ``ds``, ``score`` y ``scorable``.

        Returns
        -------
        FittedThresholder
            Umbralizador calibrado.
        """
        ...


@runtime_checkable
class FittedThresholder(Protocol):
    """Umbralizador calibrado."""

    def threshold(self, alpha: float) -> pd.DataFrame:
        """Umbral de deteccion para un nivel de alfa dado.

        Parameters
        ----------
        alpha
            Tasa de falsos positivos objetivo.

        Returns
        -------
        pandas.DataFrame
            Columnas ``unique_id`` (``NaN`` si el umbral es global), ``alpha`` y
            ``threshold``. Se precomputa sobre una rejilla de alfa para que el
            slider de la app sea una busqueda en tabla y no un recalculo, que
            estaria prohibido por la regla de que la app no calcula.
        """
        ...
