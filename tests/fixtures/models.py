"""Modelos sonda que implementan `Forecaster` para ejercitar el motor.

No son baselines ni sustituyen a `chronolab.models`: son instrumentos. Cada uno
existe para hacer observable una propiedad del motor que de otro modo solo se
podria afirmar:

- `ScaledExogProbe` **registra lo que ha visto** —el ultimo instante de su
  entrenamiento, las columnas de la trama de exogenas futuras y los estadisticos
  de su escalador— de modo que un test pueda comprobar que no vio nada posterior
  al cutoff en lugar de confiar en que asi fue.
- `SeasonalNaiveProbe` da predicciones deterministas calculables a mano.
- `FailingProbe`, `CutoffViolatingProbe` y `CrossedQuantileProbe` producen los
  tres fallos que el motor tiene que tratar de tres maneras distintas: registrar,
  abortar y reparar.
- `PartialQuantileProbe` deja sin calcular los cuantiles centrales (como hacen
  los adaptadores de statsforecast cuando solo se calibran un par de niveles):
  comprueba que reparar el cruce de los cuantiles que si existen no pisa con
  `NaN` los que no se pidieron ajustar.

Los modelos son `frozen`: el estado ajustado vive en el objeto que devuelve
`fit`. Las listas de registro se mutan, no se reasignan, que es lo que permite
que un `Forecaster` inmutable acumule observaciones a lo largo de un run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from chronolab.errors import MissingFutrExog
from chronolab.models.protocols import QUANTILES, ModelRequirements, quantile_column
from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId

SCALED_EXOG_ID = ModelId("probe_scaled_exog")
SEASONAL_NAIVE_ID = ModelId("probe_seasonal_naive")
FAILING_ID = ModelId("probe_failing")
CUTOFF_VIOLATING_ID = ModelId("probe_cutoff_violating")
CROSSED_QUANTILE_ID = ModelId("probe_crossed_quantiles")
PARTIAL_QUANTILE_ID = ModelId("probe_partial_quantiles")

EXOG_REQUIREMENTS = ModelRequirements(needs_futr_exog=True, min_context=2)
SEASONAL_REQUIREMENTS = ModelRequirements(min_context=24)
PLAIN_REQUIREMENTS = ModelRequirements()
QUANTILE_REQUIREMENTS = ModelRequirements(supports_quantiles=True)


@dataclass(frozen=True, slots=True)
class FitRecord:
    """Lo que un modelo sonda vio al ajustarse en una ventana."""

    train_first_ds: pd.Timestamp
    train_last_ds: pd.Timestamp
    train_columns: tuple[str, ...]
    n_rows: int
    scaler: dict[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class PredictRecord:
    """Lo que un modelo sonda recibio al predecir en una ventana."""

    cutoff: pd.Timestamp
    futr_columns: tuple[str, ...]
    futr_first_ds: pd.Timestamp | None
    futr_last_ds: pd.Timestamp | None


def _series_scaler(values: np.ndarray) -> tuple[float, float]:
    """Media y desviacion tipica de una serie, ignorando huecos."""
    observed = values[~np.isnan(values)]
    if observed.size == 0:
        return 0.0, 1.0
    std = float(observed.std())
    return float(observed.mean()), std if std > 0 else 1.0


def _design(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Matriz de diseno con intercepto y las columnas indicadas."""
    intercept = np.ones((len(frame), 1))
    if not columns:
        return intercept
    return np.column_stack([intercept, frame[list(columns)].to_numpy(dtype=float)])


@dataclass(frozen=True, slots=True)
class _FittedScaledExog:
    """Ajuste de `ScaledExogProbe`: escalador por serie y coeficientes."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    exog: tuple[str, ...]
    scaler: dict[str, tuple[float, float]]
    coefficients: dict[str, np.ndarray]
    owner: ScaledExogProbe

    @property
    def n_params(self) -> int | None:
        return sum(int(coef.size) for coef in self.coefficients.values())

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        if futr is None:
            raise MissingFutrExog(f"{self.model_id} necesita exogenas futuras")

        frame = futr.df
        self.owner.predictions.append(
            PredictRecord(
                cutoff=self.cutoff,
                futr_columns=tuple(frame.columns),
                futr_first_ds=None if frame.empty else pd.Timestamp(frame["ds"].min()),
                futr_last_ds=None if frame.empty else pd.Timestamp(frame["ds"].max()),
            )
        )

        parts: list[pd.DataFrame] = []
        for uid, group in frame.groupby("unique_id", sort=False):
            key = str(uid)
            mean, std = self.scaler[key]
            predicted = _design(group, self.exog) @ self.coefficients[key]
            parts.append(
                pd.DataFrame(
                    {
                        "unique_id": key,
                        "ds": group["ds"].to_numpy(),
                        "y_hat": mean + std * predicted,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)


@dataclass(frozen=True)
class ScaledExogProbe:
    """Regresion lineal sobre las exogenas futuras, con escalado ajustado en `fit`.

    El escalador por serie y los coeficientes se estiman **dentro de `fit`**, es
    decir sobre el panel que el motor entrega, que es el del tramo de
    entrenamiento de la ventana en curso. El modelo usa exactamente las columnas
    que le llegan en el `FutrFrame`: si una exogena esta declarada `hist_exog`, no
    aparece en esa trama y el modelo no puede usarla ni queriendo, que es la
    propiedad que los tests de canario miden en los dos sentidos.
    """

    model_id: ModelId = SCALED_EXOG_ID
    requires: ModelRequirements = EXOG_REQUIREMENTS
    fits: list[FitRecord] = field(default_factory=list)
    predictions: list[PredictRecord] = field(default_factory=list)

    def fit(self, train: Panel, *, h: int) -> _FittedScaledExog:
        spec = train.spec
        exog = tuple(spec.futr_exog)
        scaler: dict[str, tuple[float, float]] = {}
        coefficients: dict[str, np.ndarray] = {}

        for uid, group in train.df.groupby("unique_id", sort=False):
            key = str(uid)
            target = group[spec.target].to_numpy(dtype=float)
            mean, std = _series_scaler(target)
            scaler[key] = (mean, std)

            observed = group.loc[~np.isnan(target)]
            scaled = (target[~np.isnan(target)] - mean) / std
            design = _design(observed, exog)
            coefficients[key] = np.linalg.lstsq(design, scaled, rcond=None)[0]

        self.fits.append(
            FitRecord(
                train_first_ds=train.first_ds,
                train_last_ds=train.last_ds,
                train_columns=tuple(train.df.columns),
                n_rows=len(train.df),
                scaler=dict(scaler),
            )
        )
        return _FittedScaledExog(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=0.0,
            exog=exog,
            scaler=scaler,
            coefficients=coefficients,
            owner=self,
        )


@dataclass(frozen=True, slots=True)
class _FittedSeasonalNaive:
    """Ajuste de `SeasonalNaiveProbe`: la ultima estacion completa por serie."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    season: int
    history: dict[str, np.ndarray]

    @property
    def n_params(self) -> int | None:
        return None

    def _horizon(self, futr: FutrFrame | None, ids: Sequence[str]) -> pd.DataFrame:
        """Instantes a predecir: los del `FutrFrame` si lo hay, o `cutoff + 1..h`."""
        if futr is not None:
            return futr.df[["unique_id", "ds"]].copy()
        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:]
        return pd.DataFrame(
            {
                "unique_id": np.repeat(list(ids), self.h),
                "ds": np.tile(grid.to_numpy(), len(ids)),
            }
        )

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        horizon = self._horizon(futr, sorted(self.history))
        grid = pd.date_range(self.cutoff, horizon["ds"].max(), freq=self.freq)
        lead = grid.get_indexer(pd.DatetimeIndex(horizon["ds"]))

        values = np.empty(len(horizon), dtype=float)
        for key, history in self.history.items():
            rows = (horizon["unique_id"].astype(str) == key).to_numpy()
            values[rows] = history[(lead[rows] - 1) % self.season]

        horizon["y_hat"] = values
        return horizon


@dataclass(frozen=True)
class SeasonalNaiveProbe:
    """Naive estacional en numpy: predice el valor de hace `season` pasos.

    Determinista y calculable a mano, que es lo que permite comprobar que el
    motor alinea cada prediccion con el instante correcto en lugar de con el de
    al lado.
    """

    season: int = 24
    model_id: ModelId = SEASONAL_NAIVE_ID
    requires: ModelRequirements = SEASONAL_REQUIREMENTS

    def fit(self, train: Panel, *, h: int) -> _FittedSeasonalNaive:
        history: dict[str, np.ndarray] = {}
        for uid, group in train.df.groupby("unique_id", sort=False):
            target = group[train.spec.target].ffill().to_numpy(dtype=float)
            history[str(uid)] = target[-self.season :]
        return _FittedSeasonalNaive(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=0.0,
            freq=train.spec.freq,
            season=self.season,
            history=history,
        )


@dataclass(frozen=True, slots=True)
class _FittedFailing:
    """Ajuste que revienta al predecir en los cutoffs indicados."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    ids: tuple[str, ...]
    fail_on: tuple[pd.Timestamp, ...]

    @property
    def n_params(self) -> int | None:
        return None

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        if not self.fail_on or self.cutoff in self.fail_on:
            raise RuntimeError("el modelo ha reventado")

        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:]
        return pd.DataFrame(
            {
                "unique_id": np.repeat(list(self.ids), self.h),
                "ds": np.tile(grid.to_numpy(), len(self.ids)),
                "y_hat": 0.0,
            }
        )


@dataclass(frozen=True)
class FailingProbe:
    """Modelo que falla al predecir en las ventanas cuyo cutoff se le indique.

    Con `fail_on` vacio falla en todas. Sirve para comprobar que un fallo ocupa
    una fila con ``status="failed"`` en lugar de desaparecer del leaderboard.
    """

    fail_on: tuple[pd.Timestamp, ...] = ()
    model_id: ModelId = FAILING_ID
    requires: ModelRequirements = PLAIN_REQUIREMENTS

    def fit(self, train: Panel, *, h: int) -> _FittedFailing:
        return _FittedFailing(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=0.0,
            freq=train.spec.freq,
            ids=tuple(str(uid) for uid in train.ids()),
            fail_on=self.fail_on,
        )


@dataclass(frozen=True, slots=True)
class _FittedCutoffViolating:
    """Ajuste que predice un instante que ya conocia."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    ids: tuple[str, ...]

    @property
    def n_params(self) -> int | None:
        return None

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        # Empieza en el propio cutoff: el off-by-one clasico.
        grid = pd.date_range(self.cutoff, periods=self.h, freq=self.freq)
        return pd.DataFrame(
            {
                "unique_id": np.repeat(list(self.ids), self.h),
                "ds": np.tile(grid.to_numpy(), len(self.ids)),
                "y_hat": 0.0,
            }
        )


@dataclass(frozen=True)
class CutoffViolatingProbe:
    """Modelo que predice desde el propio cutoff, es decir un instante ya visto."""

    model_id: ModelId = CUTOFF_VIOLATING_ID
    requires: ModelRequirements = PLAIN_REQUIREMENTS

    def fit(self, train: Panel, *, h: int) -> _FittedCutoffViolating:
        return _FittedCutoffViolating(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=0.0,
            freq=train.spec.freq,
            ids=tuple(str(uid) for uid in train.ids()),
        )


@dataclass(frozen=True, slots=True)
class _FittedCrossedQuantiles:
    """Ajuste que devuelve los cuantiles al reves."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    ids: tuple[str, ...]

    @property
    def n_params(self) -> int | None:
        return None

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:]
        frame = pd.DataFrame(
            {
                "unique_id": np.repeat(list(self.ids), self.h),
                "ds": np.tile(grid.to_numpy(), len(self.ids)),
                "y_hat": 100.0,
            }
        )
        # Cuantiles en orden decreciente: el cruce que el motor tiene que reparar.
        for position, quantile in enumerate(quantiles):
            frame[quantile_column(quantile)] = 100.0 + len(quantiles) - position
        return frame


@dataclass(frozen=True)
class CrossedQuantileProbe:
    """Modelo probabilistico cuyos cuantiles salen cruzados en todas las filas."""

    model_id: ModelId = CROSSED_QUANTILE_ID
    requires: ModelRequirements = QUANTILE_REQUIREMENTS

    def fit(self, train: Panel, *, h: int) -> _FittedCrossedQuantiles:
        return _FittedCrossedQuantiles(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=0.0,
            freq=train.spec.freq,
            ids=tuple(str(uid) for uid in train.ids()),
        )


@dataclass(frozen=True, slots=True)
class _FittedPartialQuantiles:
    """Ajuste que solo calibra dos niveles y deja el resto de cuantiles sin calcular."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    ids: tuple[str, ...]

    @property
    def n_params(self) -> int | None:
        return None

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:]
        frame = pd.DataFrame(
            {
                "unique_id": np.repeat(list(self.ids), self.h),
                "ds": np.tile(grid.to_numpy(), len(self.ids)),
                "y_hat": 100.0,
            }
        )
        # Solo los dos cuantiles mas extremos, y cruzados a proposito; el
        # resto de la rejilla pedida (incluida la mediana) no se calcula,
        # igual que un adaptador que solo calibro un par de niveles conformales.
        ordered = sorted(quantiles)
        if len(ordered) >= 2:
            lowest, highest = ordered[0], ordered[-1]
            frame[quantile_column(lowest)] = 108.0  # el bajo, por encima del alto
            frame[quantile_column(highest)] = 92.0  # el alto, por debajo del bajo
        return frame


@dataclass(frozen=True)
class PartialQuantileProbe:
    """Modelo probabilistico que solo calibra dos cuantiles extremos, cruzados."""

    model_id: ModelId = PARTIAL_QUANTILE_ID
    requires: ModelRequirements = QUANTILE_REQUIREMENTS

    def fit(self, train: Panel, *, h: int) -> _FittedPartialQuantiles:
        return _FittedPartialQuantiles(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=0.0,
            freq=train.spec.freq,
            ids=tuple(str(uid) for uid in train.ids()),
        )
