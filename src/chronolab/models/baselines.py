"""Baselines en numpy puro: Naive, SeasonalNaive, WindowAverage, HistoricAverage y Drift.

Implementacion independiente de statsforecast a proposito: si el arnes es el
producto, el arnes necesita un patron de referencia calculable a mano (D15).
`tests/unit/models/test_baselines.py` verifica los pronosticos puntuales de
`NaiveForecaster` y `SeasonalNaiveForecaster` contra `statsforecast.Naive` y
`statsforecast.SeasonalNaive` sobre las mismas ventanas cuando la libreria esta
instalada (extra `ml`); sin ella el test se salta con
``pytest.importorskip("statsforecast")``, porque `models.baselines` no depende
de statsforecast en ningun caso —eso es justo lo que D15 pide proteger. Esta
independencia tambien protege frente a un cambio de convencion de la libreria
entre versiones: si `statsforecast.Naive` cambiase su formula de intervalos
manana, el numero de referencia de este modulo seguiria siendo el mismo.

Cuatro decisiones de diseno comunes a los cinco modelos:

1. **Los huecos internos se rellenan hacia atras (`ffill`) antes de calcular
   nada.** El invariante I3 garantiza que el primer y el ultimo punto de cada
   serie son observaciones reales —solo los huecos internos son `NaN`
   explicito—, asi que `ffill` siempre tiene de donde partir. Es la unica
   forma de imputacion admitida por la barrera L10: mira solo hacia atras.
2. **Los intervalos son gaussianos**, con la desviacion tipica de los residuos
   de un paso creciendo con el horizonte segun las formulas clasicas de
   Hyndman y Athanasopoulos (*Forecasting: Principles and Practice*, cap. 5.2),
   las mismas que usa `statsforecast` para sus baselines. Sin `scipy` en el
   nucleo del proyecto (D16 la pone en cuarentena en `adapters/` y `anomaly/`),
   el cuantil gaussiano se calcula con una aproximacion racional propia
   (`_normal_ppf`), en el mismo espiritu que las funciones de distribucion de
   `chronolab.evaluation.stats_tests`.
3. **`ds` sale del `FutrFrame` cuando lo hay, y de `cutoff + h_step * freq`
   cuando no.** Es el canal que describe `chronolab.evaluation.backtest`: un
   modelo sin exogenas futuras solo sabe a que instantes predecir a traves de
   esa trama. Con ``gap = 0`` y sin `FutrProvider` en el run las dos vias
   coinciden; con ``gap > 0`` el run necesita un `FutrProvider`, aunque el
   panel no declare ninguna `futr_exog`, exactamente igual que cualquier otro
   modelo.
4. **`n_params` es `None` en los cinco.** No hay parametros ajustados por
   descenso de gradiente ni seleccionados por un criterio de informacion: son
   formulas cerradas sobre estadisticos directos de la serie, y forzar un
   numero (¿cero? ¿uno por el nivel?) seria una convencion sin contenido.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd

from chronolab.models.protocols import QUANTILES, ModelRequirements, quantile_column
from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId

_NAIVE_ID = ModelId("naive")
_SEASONAL_NAIVE_ID = ModelId("seasonal_naive")
_WINDOW_AVERAGE_ID = ModelId("window_average")
_HISTORIC_AVERAGE_ID = ModelId("historic_average")
_DRIFT_ID = ModelId("drift")

__all__ = [
    "DriftForecaster",
    "HistoricAverageForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "WindowAverageForecaster",
]

_BaselineKind = Literal["naive", "seasonal_naive", "window_average", "historic_average", "drift"]


# --------------------------------------------------------------------------- #
# Cuantil gaussiano sin scipy
# --------------------------------------------------------------------------- #


def _normal_ppf(p: float) -> float:
    """Cuantil de una normal estandar: la inversa de la funcion de distribucion.

    Aproximacion racional de Acklam, precision relativa de aproximadamente
    ``1.15e-9`` en ``(0, 1)``. Sobra para columnas que se persisten en
    `float32` (docs/ARCHITECTURE.md §7.4): el error de la aproximacion es
    varios ordenes de magnitud menor que el que introduce ese redondeo.

    Parameters
    ----------
    p
        Probabilidad en ``(0, 1)``.

    Returns
    -------
    float
        El valor `z` tal que ``P(Z <= z) = p`` para `Z` normal estandar.

    Raises
    ------
    ValueError
        Si `p` cae fuera de ``(0, 1)``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probabilidad fuera de (0, 1): {p}")

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        numerator = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        denominator = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return numerator / denominator
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        denominator = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        return numerator / denominator
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    numerator = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    denominator = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    return -numerator / denominator


# --------------------------------------------------------------------------- #
# Ajuste por serie
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _SeriesState:
    """Estadisticos de una serie, ajustados hasta un cutoff, para un baseline.

    Attributes
    ----------
    last
        Ultimo valor observado. Usado por `naive` y `drift`.
    drift
        Pendiente media por paso, ``(last - first) / (n - 1)``. Solo `drift`.
    level
        Nivel constante del pronostico. Usado por `window_average` y
        `historic_average`.
    season
        Ultimos `season` valores, en orden ``[t-season+1 .. t]``. Solo
        `seasonal_naive`; tupla vacia en el resto.
    sigma
        Desviacion tipica muestral de los residuos de un paso. `NaN` si no hay
        datos suficientes para estimarla —nunca un intervalo inventado.
    n_train
        Observaciones no nulas usadas en el ajuste.
    """

    last: float
    drift: float
    level: float
    season: tuple[float, ...]
    sigma: float
    n_train: int


def _series_arrays(panel: Panel) -> dict[str, np.ndarray]:
    """Valores de la objetivo por serie, con los huecos internos rellenados hacia atras.

    Parameters
    ----------
    panel
        Panel de entrenamiento, ya recortado por el motor a ``ds <= cutoff``.

    Returns
    -------
    dict
        ``unique_id -> array`` en orden cronologico. El primer y el ultimo
        valor de cada serie son observaciones reales (invariante I3); `ffill`
        solo puede dejar `NaN` si una serie entera esta vacia, y eso se
        detecta en el ajuste de cada baseline, no aqui.
    """
    target = panel.spec.target
    result: dict[str, np.ndarray] = {}
    for uid, group in panel.df.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds")[target].to_numpy(dtype=float)
        result[str(uid)] = pd.Series(ordered).ffill().to_numpy()
    return result


def _require_finite(values: np.ndarray, *, uid: str, model: str, minimum: int) -> None:
    """Comprueba que una serie tiene suficientes observaciones utilizables.

    Parameters
    ----------
    values
        Serie ya rellenada hacia atras.
    uid
        Identificador de la serie, para el mensaje.
    model
        Nombre del baseline, para el mensaje.
    minimum
        Observaciones minimas exigidas.

    Raises
    ------
    ValueError
        Si la serie tiene menos de `minimum` observaciones, o si queda algun
        `NaN` tras el `ffill` —solo posible si la serie no tiene ni una
        observacion real en el tramo de entrenamiento.
    """
    if values.size < minimum:
        raise ValueError(
            f"{model}: la serie '{uid}' tiene {values.size} observaciones y hacen "
            f"falta al menos {minimum}"
        )
    if np.isnan(values).any():
        raise ValueError(f"{model}: la serie '{uid}' no tiene ninguna observacion real")


def _sample_std(residuals: np.ndarray) -> float:
    """Desviacion tipica muestral, o `NaN` si hay menos de dos residuos.

    Parameters
    ----------
    residuals
        Residuos de un paso.

    Returns
    -------
    float
        ``std(residuals, ddof=1)``, o `NaN` si `residuals` tiene menos de dos
        elementos: con uno solo, la varianza muestral no esta definida, y
        devolver un numero inventado seria peor que no devolver ninguno.
    """
    if residuals.size < 2:
        return float("nan")
    return float(np.std(residuals, ddof=1))


def _fit_naive(values: np.ndarray) -> _SeriesState:
    """Estado de `NaiveForecaster`: ultimo valor y sigma de las diferencias de un paso."""
    diffs = np.diff(values)
    return _SeriesState(
        last=float(values[-1]),
        drift=0.0,
        level=float(values[-1]),
        season=(),
        sigma=_sample_std(diffs),
        n_train=values.size,
    )


def _fit_drift(values: np.ndarray) -> _SeriesState:
    """Estado de `DriftForecaster`: pendiente media y sigma de los residuos sobre ella."""
    n = values.size
    drift = float((values[-1] - values[0]) / (n - 1))
    residuals = np.diff(values) - drift
    return _SeriesState(
        last=float(values[-1]),
        drift=drift,
        level=float(values[-1]),
        season=(),
        sigma=_sample_std(residuals),
        n_train=n,
    )


def _fit_seasonal_naive(values: np.ndarray, season: int) -> _SeriesState:
    """Estado de `SeasonalNaiveForecaster`: la ultima estacion y sigma de los residuos estacionales."""
    residuals = values[season:] - values[:-season]
    return _SeriesState(
        last=float(values[-1]),
        drift=0.0,
        level=float(values[-1]),
        season=tuple(float(v) for v in values[-season:]),
        sigma=_sample_std(residuals),
        n_train=values.size,
    )


def _fit_window_average(values: np.ndarray, window: int) -> _SeriesState:
    """Estado de `WindowAverageForecaster`: media de las ultimas `window` observaciones."""
    windowed = values[-window:]
    level = float(np.mean(windowed))
    return _SeriesState(
        last=float(values[-1]),
        drift=0.0,
        level=level,
        season=(),
        sigma=_sample_std(windowed - level),
        n_train=values.size,
    )


def _fit_historic_average(values: np.ndarray) -> _SeriesState:
    """Estado de `HistoricAverageForecaster`: media de toda la serie de entrenamiento."""
    level = float(np.mean(values))
    return _SeriesState(
        last=float(values[-1]),
        drift=0.0,
        level=level,
        season=(),
        sigma=_sample_std(values - level),
        n_train=values.size,
    )


def _forecast_step(
    kind: _BaselineKind, state: _SeriesState, season: int, h_step: int
) -> tuple[float, float]:
    """Pronostico puntual y sigma del paso `h_step`, segun la formula del baseline.

    Las formulas de crecimiento de sigma son las clasicas de Hyndman y
    Athanasopoulos (cap. 5.2), las mismas que usa `statsforecast`:

    - `naive`: ``sigma * sqrt(h)``.
    - `drift`: ``sigma * sqrt(h * (1 + h / (n - 1)))``.
    - `seasonal_naive`: ``sigma * sqrt(k + 1)`` con ``k = (h - 1) // m``, es
      decir, el numero de ciclos estacionales completos que ya se han cruzado.
    - `window_average`: sigma constante, sin crecimiento con `h` — es la
      aproximacion mas simple y la que usa `statsforecast` para esta familia.
    - `historic_average`: ``sigma * sqrt(1 + 1/n)``, la varianza de predecir
      con una media muestral.

    Parameters
    ----------
    kind
        Tipo de baseline.
    state
        Estado ajustado de la serie.
    season
        Longitud estacional. Solo se usa si ``kind == "seasonal_naive"``.
    h_step
        Paso de prediccion, en ``[1, h]``.

    Returns
    -------
    tuple
        Pronostico puntual y sigma del paso. Sigma es `NaN` si `state.sigma`
        lo es.
    """
    if kind == "naive":
        return state.last, state.sigma * math.sqrt(h_step)
    if kind == "drift":
        y_hat = state.last + h_step * state.drift
        factor = h_step * (1.0 + h_step / max(state.n_train - 1, 1))
        return y_hat, state.sigma * math.sqrt(factor)
    if kind == "seasonal_naive":
        y_hat = state.season[(h_step - 1) % season]
        cycles = (h_step - 1) // season
        return y_hat, state.sigma * math.sqrt(cycles + 1)
    if kind == "window_average":
        return state.level, state.sigma
    if kind == "historic_average":
        return state.level, state.sigma * math.sqrt(1.0 + 1.0 / max(state.n_train, 1))
    raise ValueError(f"tipo de baseline no reconocido: {kind}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# `FittedForecaster` comun a los cinco baselines
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _FittedBaseline:
    """Ajuste comun a los cinco baselines: un `_SeriesState` por serie mas el tipo."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    kind: _BaselineKind
    season: int
    states: dict[str, _SeriesState]

    @property
    def n_params(self) -> int | None:
        """``None``: los baselines no tienen parametros ajustados por optimizacion."""
        return None

    def _horizon(self, futr: FutrFrame | None) -> dict[str, list[pd.Timestamp]]:
        """Instantes a predecir por serie: los del `FutrFrame` si lo hay, o `cutoff + h_step * freq`.

        Parameters
        ----------
        futr
            Exogenas futuras de la ventana, o ``None``. Los baselines no usan
            sus columnas —``needs_futr_exog=False``—, pero si usan su `ds`
            cuando esta disponible: es el unico canal por el que un modelo sin
            exogenas conoce el tramo exacto que se evalua (gap incluido).

        Returns
        -------
        dict
            ``unique_id -> lista de `h` marcas de tiempo``, en orden.
        """
        if futr is not None and not futr.df.empty:
            return {
                str(uid): group.sort_values("ds")["ds"].tolist()
                for uid, group in futr.df.groupby("unique_id", sort=False)
            }
        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:].tolist()
        return dict.fromkeys(self.states, grid)

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        """Predice `h` pasos para todas las series del entrenamiento.

        Parameters
        ----------
        futr
            Ignorado como fuente de exogenas —los baselines no las usan—, pero
            consultado para el `ds` exacto del tramo evaluado cuando esta
            disponible.
        quantiles
            Cuantiles a estimar. Se calculan siempre con la aproximacion
            gaussiana de `_normal_ppf`; son `NaN` donde `sigma` lo es.

        Returns
        -------
        pandas.DataFrame
            ``unique_id``, ``ds``, ``y_hat`` y una columna por cuantil.

        Raises
        ------
        ValueError
            Si el `FutrFrame` no trae exactamente `h` instantes para alguna de
            las series entrenadas.
        """
        rows: list[dict[str, object]] = []
        horizon = self._horizon(futr)

        for uid, state in self.states.items():
            instants = horizon.get(uid)
            if instants is None or len(instants) != self.h:
                found = 0 if instants is None else len(instants)
                raise ValueError(
                    f"{self.model_id}: se esperaban {self.h} instantes a predecir para "
                    f"'{uid}' y se han encontrado {found}"
                )
            for h_step, ds in enumerate(instants, start=1):
                y_hat, sigma = _forecast_step(self.kind, state, self.season, h_step)
                row: dict[str, object] = {"unique_id": uid, "ds": ds, "y_hat": y_hat}
                for quantile in quantiles:
                    row[quantile_column(quantile)] = (
                        float("nan") if math.isnan(sigma) else y_hat + sigma * _normal_ppf(quantile)
                    )
                rows.append(row)

        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Los cinco `Forecaster`
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NaiveForecaster:
    """Pronostico ingenuo: el ultimo valor observado, repetido en todo el horizonte.

    El patron de referencia minimo: cualquier modelo tiene que batirlo para
    justificar su complejidad (D15). Es tambien la base del denominador de
    MASE cuando la estacionalidad mas corta es ``1`` —para datos ya
    desestacionalizados, por ejemplo.

    Parameters
    ----------
    model_id
        Identificador del modelo.
    """

    model_id: ModelId = _NAIVE_ID

    @property
    def requires(self) -> ModelRequirements:
        """Exige al menos dos observaciones: una para el nivel, otra para estimar sigma."""
        return ModelRequirements(min_context=2, refit_cost="free")

    def fit(self, train: Panel, *, h: int) -> _FittedBaseline:
        """Ajusta el ultimo valor y la sigma de las diferencias de un paso, por serie."""
        started = perf_counter()
        arrays = _series_arrays(train)
        for uid, values in arrays.items():
            _require_finite(values, uid=uid, model="naive", minimum=self.requires.min_context)
        states = {uid: _fit_naive(values) for uid, values in arrays.items()}
        return _FittedBaseline(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=perf_counter() - started,
            freq=train.spec.freq,
            kind="naive",
            season=1,
            states=states,
        )


@dataclass(frozen=True)
class SeasonalNaiveForecaster:
    """Pronostico ingenuo estacional: el valor de hace `season` pasos.

    Parameters
    ----------
    season
        Longitud estacional en pasos, mayor o igual que uno. Por defecto 24,
        el ciclo diario de una serie horaria. Para batir a `SeasonalNaive` sin
        trampa hace falta pasarle la estacionalidad correcta del dataset
        (``PanelSpec.mase_season``, tipicamente).
    model_id
        Identificador del modelo.

    Raises
    ------
    ValueError
        Si `season` es menor que uno.
    """

    season: int = 24
    model_id: ModelId = _SEASONAL_NAIVE_ID

    def __post_init__(self) -> None:
        """Valida la longitud estacional."""
        if self.season < 1:
            raise ValueError(f"season debe ser >= 1: {self.season}")

    @property
    def requires(self) -> ModelRequirements:
        """Exige ``season + 2`` observaciones: una estacion completa y dos residuos para sigma."""
        return ModelRequirements(min_context=self.season + 2, refit_cost="free")

    def fit(self, train: Panel, *, h: int) -> _FittedBaseline:
        """Ajusta la ultima estacion completa y la sigma de los residuos estacionales."""
        started = perf_counter()
        arrays = _series_arrays(train)
        for uid, values in arrays.items():
            _require_finite(values, uid=uid, model="seasonal_naive", minimum=self.season)
        states = {uid: _fit_seasonal_naive(values, self.season) for uid, values in arrays.items()}
        return _FittedBaseline(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=perf_counter() - started,
            freq=train.spec.freq,
            kind="seasonal_naive",
            season=self.season,
            states=states,
        )


@dataclass(frozen=True)
class WindowAverageForecaster:
    """Media movil de las ultimas `window` observaciones, constante en todo el horizonte.

    Parameters
    ----------
    window
        Longitud de la ventana en pasos, mayor o igual que uno.
    model_id
        Identificador del modelo.

    Raises
    ------
    ValueError
        Si `window` es menor que uno.
    """

    window: int = 24
    model_id: ModelId = _WINDOW_AVERAGE_ID

    def __post_init__(self) -> None:
        """Valida la longitud de la ventana."""
        if self.window < 1:
            raise ValueError(f"window debe ser >= 1: {self.window}")

    @property
    def requires(self) -> ModelRequirements:
        """Exige `window` observaciones para poder promediarlas."""
        return ModelRequirements(min_context=self.window, refit_cost="free")

    def fit(self, train: Panel, *, h: int) -> _FittedBaseline:
        """Ajusta la media de las ultimas `window` observaciones, por serie."""
        started = perf_counter()
        arrays = _series_arrays(train)
        for uid, values in arrays.items():
            _require_finite(values, uid=uid, model="window_average", minimum=self.window)
        states = {uid: _fit_window_average(values, self.window) for uid, values in arrays.items()}
        return _FittedBaseline(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=perf_counter() - started,
            freq=train.spec.freq,
            kind="window_average",
            season=1,
            states=states,
        )


@dataclass(frozen=True)
class HistoricAverageForecaster:
    """Media de toda la historia de entrenamiento, constante en todo el horizonte.

    Parameters
    ----------
    model_id
        Identificador del modelo.
    """

    model_id: ModelId = _HISTORIC_AVERAGE_ID

    @property
    def requires(self) -> ModelRequirements:
        """Exige al menos una observacion."""
        return ModelRequirements(min_context=1, refit_cost="free")

    def fit(self, train: Panel, *, h: int) -> _FittedBaseline:
        """Ajusta la media de toda la serie de entrenamiento, por serie."""
        started = perf_counter()
        arrays = _series_arrays(train)
        for uid, values in arrays.items():
            _require_finite(values, uid=uid, model="historic_average", minimum=1)
        states = {uid: _fit_historic_average(values) for uid, values in arrays.items()}
        return _FittedBaseline(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=perf_counter() - started,
            freq=train.spec.freq,
            kind="historic_average",
            season=1,
            states=states,
        )


@dataclass(frozen=True)
class DriftForecaster:
    """Naive con deriva: extrapola la pendiente media entre el primer y el ultimo punto.

    ``y_hat(h) = last + h * (last - first) / (n - 1)``. Bate al naive simple en
    series con tendencia y lo iguala cuando no la hay.

    Parameters
    ----------
    model_id
        Identificador del modelo.
    """

    model_id: ModelId = _DRIFT_ID

    @property
    def requires(self) -> ModelRequirements:
        """Exige al menos dos observaciones: una pendiente necesita dos puntos."""
        return ModelRequirements(min_context=2, refit_cost="free")

    def fit(self, train: Panel, *, h: int) -> _FittedBaseline:
        """Ajusta la pendiente media y la sigma de los residuos sobre ella, por serie."""
        started = perf_counter()
        arrays = _series_arrays(train)
        for uid, values in arrays.items():
            _require_finite(values, uid=uid, model="drift", minimum=2)
        states = {uid: _fit_drift(values) for uid, values in arrays.items()}
        return _FittedBaseline(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=perf_counter() - started,
            freq=train.spec.freq,
            kind="drift",
            season=1,
            states=states,
        )
