"""Discords con stumpy: deteccion robusta y sin entrenamiento.

Un discord es la subsecuencia de longitud `m` mas distinta de cualquier otra
subsecuencia de la misma serie: la distancia (euclidiana, z-normalizada) a su
vecino mas cercano, que es exactamente lo que `stumpy.stump` calcula para
**toda** posicion de una serie de una sola pasada. No hay modelo que ajustar,
no hay hiperparametro que aprender de los datos -solo `m`, que se elige, no se
estima-, y por eso es la referencia de contraste del arnes: si un detector
calibrado no bate a Matrix Profile, calibrar no esta pagando su coste.

**`needs_calibration=False`, tal cual lo nombra el protocolo.** `fit` no
ajusta nada: solo fija el `cutoff` y conserva el tramo final de `calib`
-`m - 1` puntos- que hace falta para que la primera ventana de `score` tenga
contexto completo, exactamente igual que el calentamiento de cualquier
detector de ventana se cuenta desde el principio del panel y no desde el
cutoff de cada tramo puntuado. Eso no es calibrar: es la misma mecanica que
sostiene `requires.window` en cualquier detector, aplicada a uno sin
parametro que aprender. Por eso este detector no emite `severity`, `calib_n`
ni `side`: no hay nocion de umbral calibrado sobre la que basarlos
(docs/ARCHITECTURE.md §5.3).

**Puntuacion retrospectiva, no en linea.** `score` calcula un unico perfil de
matriz sobre el tramo completo que se le pasa (mas el contexto prestado de la
calibracion): no hay estado que avanzar punto a punto, porque el vecino mas
cercano de una subsecuencia puede estar en cualquier parte del tramo, delante
o detras. Es correcto porque `ScoringFrame` son predicciones **ya realizadas**
-un `score` audita un tramo cerrado del pasado, no predice el futuro-, pero
hace que este detector, a diferencia del conformal, no sirva para puntuar
"segun van llegando" datos sin volver a calcular el tramo entero.

**Score comparable, con una salvedad declarada.** Se aplica la misma
transformacion ``-log10(p)`` que el resto del arnes -la exige
`chronolab.anomaly.events.aggregate_events`, que no distingue detectores-,
pero el pool de referencia no es un tramo de calibracion disjunto: es la
propia distribucion de distancias del tramo que se esta puntuando, por serie.
Es la unica opcion sin ajuste, y es honesta mientras la mayoria del tramo sea
normal -lo habitual-, pero **no** lleva la garantia de tasa de falsos
positivos de un detector calibrado: un tramo mayoritariamente anomalo
inflaria su propio pool de referencia y aplanaria el score. Se documenta en
vez de fingir la misma garantia que `chronolab.anomaly.isolation` o
`chronolab.anomaly.conformal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from chronolab.anomaly.protocols import DetectorRequirements, ScoringFrame
from chronolab.errors import CutoffViolation
from chronolab.panel import PanelSpec
from chronolab.types import DetectorId

__all__ = ["SCORE_COLUMNS", "FittedMatrixProfileDetector", "MatrixProfileDetector"]

SCORE_COLUMNS: tuple[str, ...] = ("unique_id", "ds", "score", "scorable")
"""Columnas que devuelve `FittedMatrixProfileDetector.score`.

Sin `severity`, `calib_n` ni `side`: `needs_calibration=False` y el protocolo
declara que un detector sin nocion de umbral calibrado no las emite
(docs/ARCHITECTURE.md §5.3).
"""


def _require_stumpy() -> Any:
    """Importa `stumpy` bajo demanda.

    Returns
    -------
    Any
        El modulo `stumpy`. Tipo `Any` a proposito: `stumpy` esta en la
        cuarentena de tipos de docs/ARCHITECTURE.md D16.

    Raises
    ------
    ImportError
        Si `stumpy` no esta instalado.
    """
    try:
        import stumpy
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra ml
        raise ImportError(
            "chronolab.anomaly.matrix_profile necesita el extra 'ml': `uv sync --extra ml`."
        ) from exc
    return stumpy


@dataclass(frozen=True, slots=True)
class MatrixProfileDetector:
    """Configuracion del detector. Inmutable y sin estado, salvo el cutoff tras `fit`.

    Parameters
    ----------
    m
        Longitud de la subsecuencia del perfil de matriz. Es una eleccion,
        no un ajuste: tipicamente una fraccion de la estacionalidad mas
        corta del panel (por ejemplo, unas horas dentro de un ciclo diario).

    Raises
    ------
    ValueError
        Si `m` es menor que 3, el minimo que admite `stumpy`.
    """

    m: int = 24

    def __post_init__(self) -> None:
        """Valida la configuracion."""
        if self.m < 3:
            raise ValueError(f"m debe ser >= 3: {self.m}")

    @property
    def detector_id(self) -> DetectorId:
        """Identificador estable. Clave de particion de `anomaly_scores`."""
        return DetectorId(f"matrix_profile_m{self.m:03d}")

    @property
    def requires(self) -> DetectorRequirements:
        """Necesidades declaradas.

        ``needs_calibration=False``: es el ejemplo que nombra el protocolo
        (docs/ARCHITECTURE.md §5.3) para un metodo sin ajuste.
        """
        return DetectorRequirements(
            needs_forecast=False,
            needs_quantiles=False,
            window=self.m,
            needs_calibration=False,
            fit_cost="free",
        )

    def fit(self, calib: ScoringFrame) -> FittedMatrixProfileDetector:
        """Fija el cutoff y conserva el contexto minimo para el primer `score`.

        No ajusta ningun parametro ni estadistico: `m` es una eleccion de
        `self`, no algo que se estime aqui. Lo unico que se guarda es el
        tramo final de `calib`, tan largo como hace falta para que la
        primera subsecuencia de la siguiente llamada a `score` tenga
        contexto completo.

        Parameters
        ----------
        calib
            Tramo de calibracion.

        Returns
        -------
        FittedMatrixProfileDetector
            Con ``cutoff = calib.end``.

        Raises
        ------
        ValueError
            Si falta la columna objetivo.
        """
        if "y" not in calib.df.columns:
            raise ValueError("el tramo de calibracion no trae la columna 'y'")
        tails = _tails(calib.df, m=self.m)
        return FittedMatrixProfileDetector(
            detector=self, spec=calib.spec, cutoff=calib.end, tails=tails
        )


@dataclass(frozen=True, slots=True)
class FittedMatrixProfileDetector:
    """Detector "calibrado" hasta un instante concreto -en el sentido minimo de §5.3.

    Attributes
    ----------
    detector
        Configuracion de la que procede.
    spec
        Especificacion del panel.
    cutoff
        Ultimo instante usado en la calibracion. `score` exige ``ds > cutoff``.
    tails
        Ultimas ``m - 1`` filas de `calib.df` por serie, para el contexto del
        primer `score`.
    """

    detector: MatrixProfileDetector
    spec: PanelSpec
    cutoff: pd.Timestamp
    tails: dict[str, pd.DataFrame]

    @property
    def detector_id(self) -> DetectorId:
        """Identificador del detector del que procede."""
        return self.detector.detector_id

    def score(self, frame: ScoringFrame) -> pd.DataFrame:
        """Puntua cada marca de tiempo del tramo con un unico perfil de matriz por serie.

        Parameters
        ----------
        frame
            Tramo a puntuar, con ``frame.start > cutoff``.

        Returns
        -------
        pandas.DataFrame
            Columnas `SCORE_COLUMNS`.

        Raises
        ------
        CutoffViolation
            Si `frame` empieza en un instante que la calibracion ya conocia.
        """
        if frame.start <= self.cutoff:
            raise CutoffViolation(
                f"{self.detector_id} esta calibrado hasta {self.cutoff} y se le pide "
                f"puntuar desde {frame.start}, anterior o igual"
            )

        stumpy = _require_stumpy()
        m = self.detector.m
        out = frame.df[["unique_id", "ds"]].copy()
        out["unique_id"] = out["unique_id"].astype(str)

        per_series: list[pd.DataFrame] = []
        for uid, own in frame.df.groupby("unique_id", sort=False):
            key = str(uid)
            own_sorted = own.sort_values("ds")
            tail = self.tails.get(key)
            n_tail = 0 if tail is None else len(tail)
            values = np.concatenate(
                [
                    tail["y"].to_numpy(dtype=float) if tail is not None else np.empty(0),
                    own_sorted["y"].to_numpy(dtype=float),
                ]
            )
            raw = _matrix_profile_distance(stumpy, values, m=m)
            uid_score = np.full(len(own_sorted), np.nan, dtype=np.float32)
            uid_scorable = np.zeros(len(own_sorted), dtype=bool)

            if raw is not None:
                # `raw[i]` es la distancia de la subsecuencia que empieza en
                # `i`; se asigna a su ultimo punto, `i + m - 1`, que es cuando
                # esa subsecuencia queda completamente observada. Solo
                # interesan las posiciones que caen dentro del tramo propio,
                # no en el contexto prestado de la calibracion.
                end_positions = np.arange(raw.size) + m - 1
                own_mask = end_positions >= n_tail
                own_positions = end_positions[own_mask] - n_tail
                own_raw = raw[own_mask]
                valid = np.isfinite(own_raw)
                if valid.any():
                    pool = own_raw[valid]
                    uid_score[own_positions[valid]] = _self_referential_score(pool, pool)
                    uid_scorable[own_positions[valid]] = True

            per_series.append(
                pd.DataFrame(
                    {
                        "unique_id": key,
                        "ds": own_sorted["ds"].to_numpy(),
                        "score": uid_score,
                        "scorable": uid_scorable,
                    }
                )
            )

        computed = (
            pd.concat(per_series, ignore_index=True)
            if per_series
            else pd.DataFrame(columns=["unique_id", "ds", "score", "scorable"])
        )
        merged = out.merge(computed, on=["unique_id", "ds"], how="left", validate="one_to_one")
        merged["score"] = merged["score"].astype(np.float32)
        merged["scorable"] = merged["scorable"].fillna(False).astype(bool)
        return merged[list(SCORE_COLUMNS)]


def _matrix_profile_distance(stumpy: Any, values: np.ndarray, *, m: int) -> np.ndarray | None:
    """Distancia al vecino mas cercano de cada subsecuencia de longitud `m`.

    Parameters
    ----------
    stumpy
        Modulo `stumpy` ya importado.
    values
        Serie completa (contexto prestado + tramo a puntuar), en orden
        temporal. Puede contener `NaN` donde el panel tiene huecos.
    m
        Longitud de la subsecuencia.

    Returns
    -------
    numpy.ndarray or None
        Columna 0 de la salida de `stumpy.stump`, como `float64`. ``None`` si
        la serie es demasiado corta para admitir ninguna subsecuencia
        comparable -`stumpy` exige al menos ``2m + 1`` puntos-, en cuyo caso
        el tramo queda entero como no puntuable en vez de fallar.
    """
    if values.size < 2 * m + 1:
        return None
    profile = stumpy.stump(values, m)
    return np.asarray(profile[:, 0], dtype=np.float64)


def _self_referential_score(raw: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """P-valor conformal de cada distancia contra el pool de su propia serie y tramo.

    Misma formula que `chronolab.anomaly.isolation.pool_score`, repetida en
    vez de importada porque aqui el pool no es un tramo de calibracion
    disjunto: es el propio tramo que se puntua (ver el docstring del modulo).
    Mezclar ambos detras de una funcion compartida disfrazaria esa diferencia.

    Parameters
    ----------
    raw
        Distancias a puntuar.
    pool
        Distancias de referencia -aqui, las mismas `raw` de esa serie en este
        tramo-.

    Returns
    -------
    numpy.ndarray
        ``-log10(p)`` en `float32`.
    """
    sorted_pool = np.sort(pool)
    count_ge = pool.size - np.searchsorted(sorted_pool, raw, side="left")
    p_value = (1.0 + count_ge) / (pool.size + 1.0)
    return (-np.log10(p_value)).astype(np.float32)


def _tails(df: pd.DataFrame, *, m: int) -> dict[str, pd.DataFrame]:
    """Ultimas `m - 1` filas de cada serie, para el contexto del primer `score`.

    Parameters
    ----------
    df
        Trama de la que extraer la cola. Tipicamente `calib.df`.
    m
        Longitud de la subsecuencia del detector.

    Returns
    -------
    dict
        ``unique_id -> ultimas filas``, ordenadas por `ds`.
    """
    needed = m - 1
    tails: dict[str, pd.DataFrame] = {}
    if needed <= 0:
        return tails
    for uid, group in df.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds")
        tails[str(uid)] = ordered.tail(needed).reset_index(drop=True)
    return tails
