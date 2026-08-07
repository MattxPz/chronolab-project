"""`Thresholder`: convierte scores continuos en etiquetas para una rejilla de alfa.

Separar puntuar de umbralizar es obligatorio: VUS-PR necesita el score continuo
y F1 por rangos necesita etiquetas. Si el detector devolviera etiquetas se
perderia irreversiblemente lo que exige la metrica principal.

Para el detector conformal el umbral es ``-log10(alpha)``, igual para toda serie
y todo grupo, porque toda la calibracion vive dentro del score. Que esta tabla
quede casi vacia no es un desperdicio: es la comprobacion visible de que ese
score si esta calibrado, frente a IsolationForest o el autoencoder, cuyos
umbrales habra que estimar como cuantiles empiricos.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from chronolab.anomaly.conformal import ALPHA_GRID

__all__ = ["THRESHOLD_COLUMNS", "ConformalThresholder", "FittedConformalThresholder"]

THRESHOLD_COLUMNS: tuple[str, ...] = ("unique_id", "alpha", "threshold", "reachable")
"""Columnas de la tabla de umbrales (docs/ARCHITECTURE.md §7.4, mas `reachable`)."""


@dataclass(frozen=True, slots=True)
class ConformalThresholder:
    """Umbralizador del detector conformal.

    Parameters
    ----------
    alpha_grid
        Rejilla de alfa para la que se precomputan umbrales. Precomputarla es lo
        que convierte el deslizador de la app en una busqueda en tabla en lugar
        de un recalculo, que estaria prohibido por A5.

    Raises
    ------
    ValueError
        Si la rejilla esta vacia, no es creciente o se sale de ``(0, 1)``.
    """

    alpha_grid: tuple[float, ...] = ALPHA_GRID

    def __post_init__(self) -> None:
        """Valida la rejilla de alfa."""
        if not self.alpha_grid:
            raise ValueError("alpha_grid no puede estar vacia")
        if list(self.alpha_grid) != sorted(set(self.alpha_grid)):
            raise ValueError(f"alpha_grid debe ser estrictamente creciente: {self.alpha_grid}")
        if not all(0.0 < value < 1.0 for value in self.alpha_grid):
            raise ValueError(f"alpha_grid fuera de (0, 1): {self.alpha_grid}")

    def fit(self, calib_scores: pd.DataFrame) -> "FittedConformalThresholder":
        """Registra hasta donde llega la resolucion del score.

        No estima el umbral: el umbral se conoce en forma cerrada y estimarlo
        anadiria error de estimacion a un numero exacto. Lo que si hace falta
        mirar en los scores es **hasta donde llegan**: un p-valor conformal no
        puede bajar de ``1 / (n + 1)``, asi que por debajo de cierto alfa no hay
        cola observada. Marcarlo es preferible a devolver un umbral que ningun
        punto podra cruzar nunca sin que nada lo advierta.

        Parameters
        ----------
        calib_scores
            Columnas ``unique_id``, ``ds``, ``score`` y ``scorable``.

        Returns
        -------
        FittedConformalThresholder
            Con el techo de score observado en calibracion.

        Raises
        ------
        ValueError
            Si faltan columnas obligatorias.
        """
        missing = {"score", "scorable"} - set(calib_scores.columns)
        if missing:
            raise ValueError(f"faltan columnas obligatorias en los scores: {sorted(missing)}")

        usable = calib_scores.loc[calib_scores["scorable"].astype(bool), "score"]
        values = usable.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        ceiling = float(finite.max()) if finite.size else math.inf
        return FittedConformalThresholder(alpha_grid=self.alpha_grid, score_ceiling=ceiling)


@dataclass(frozen=True, slots=True)
class FittedConformalThresholder:
    """Umbralizador calibrado.

    Attributes
    ----------
    alpha_grid
        Rejilla precomputada.
    score_ceiling
        Score maximo observado en calibracion. Un alfa cuyo umbral lo supere es
        inalcanzable con esa muestra: se devuelve marcado, nunca extrapolado.
    """

    alpha_grid: tuple[float, ...]
    score_ceiling: float

    def threshold(self, alpha: float) -> pd.DataFrame:
        """Umbral de deteccion para un nivel de alfa dado.

        Parameters
        ----------
        alpha
            Tasa de falsos positivos objetivo.

        Returns
        -------
        pandas.DataFrame
            Una fila, con ``unique_id`` a nulo porque el umbral es global: la
            calibracion condicional por hora y adelanto ya esta dentro del score.

        Raises
        ------
        ValueError
            Si `alpha` cae fuera de ``(0, 1)``.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha fuera de (0, 1): {alpha}")
        return self._frame([alpha])

    def table(self) -> pd.DataFrame:
        """Tabla completa de umbrales sobre la rejilla precomputada.

        Returns
        -------
        pandas.DataFrame
            Una fila por alfa de la rejilla, en orden creciente.
        """
        return self._frame(list(self.alpha_grid))

    def _frame(self, alphas: list[float]) -> pd.DataFrame:
        """Construye la tabla de umbrales de una lista de alfas.

        Parameters
        ----------
        alphas
            Niveles a tabular.

        Returns
        -------
        pandas.DataFrame
            Con las columnas `THRESHOLD_COLUMNS`.
        """
        thresholds = np.array([-math.log10(value) for value in alphas], dtype=np.float32)
        return pd.DataFrame(
            {
                "unique_id": pd.Series([None] * len(alphas), dtype="object"),
                "alpha": np.asarray(alphas, dtype=np.float32),
                "threshold": thresholds,
                "reachable": thresholds <= np.float32(self.score_ceiling),
            }
        )[list(THRESHOLD_COLUMNS)]
