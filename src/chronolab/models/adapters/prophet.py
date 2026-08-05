"""Adaptador de Prophet: un ajuste por serie, con festivos y regresores exogenos.

Coste O(n_series) por ventana; es el principal motivo del riesgo R2 y de que el
dataset de evaluacion sea un subconjunto curado.

Prophet no tiene un modo multi-serie: `ProphetForecaster.fit` ajusta un
`prophet.Prophet()` independiente por cada `unique_id` del panel de
entrenamiento, con estacionalidad diaria, semanal y anual, festivos del pais
configurado y la temperatura como regresor adicional (`add_regressor`). El
regresor tiene que conocerse a futuro para poder predecir con el —es
exactamente la definicion de `futr_exog`—, asi que `requires.needs_futr_exog`
es `True` cuando hay algun regresor configurado, y `predict` exige un
`FutrFrame` igual que cualquier otro modelo que declare esa necesidad.

Import perezoso
----------------
`prophet` vive en el extra `ml`, no en el nucleo. Igual que en
`adapters/statsforecast.py`, el import real ocurre dentro de
`_require_prophet()`, llamada solo al ajustar, para que el modulo se pueda
importar sin el extra (lo exige `tests/unit/test_module_tree.py` en el
entorno por defecto de CI).

Cuantiles via muestras posteriores, no via `interval_width`
------------------------------------------------------------
Prophet expone un unico `interval_width` por llamada a `predict`, pensado para
un intervalo simetrico. Para cubrir la rejilla canonica de siete cuantiles del
proyecto de una sola vez, este adaptador usa `Prophet.predictive_samples`, que
devuelve muestras de la distribucion predictiva completa (tendencia +
estacionalidad + regresores, con su incertidumbre), y calcula cada cuantil
pedido como el percentil empirico de esas muestras. El pronostico puntual
(`y_hat`) sigue siendo el de `Prophet.predict`, no la mediana muestral: son
casi iguales, pero `predict` es determinista y la muestra no.

Prophet no acepta una semilla en su constructor, asi que el muestreo posterior
no es reproducible por defecto entre ejecuciones. Este adaptador fija la
semilla global de numpy (`np.random.seed(seed)`) justo antes de cada llamada a
`predictive_samples`, que es el mismo mecanismo que usa la propia libreria
internamente para muestrear: no es una garantia tan fuerte como un generador
propio, pero hace reproducible un run que no comparte proceso con otro codigo
que tambien consuma el generador global de numpy en paralelo.

`ds` en UTC ingenuo: por que no hace falta convertir a hora local
--------------------------------------------------------------------
El panel almacena `ds` en UTC ingenuo (invariante I2), no en la hora local de
`spec.tz_display`. Prophet extrae "hora del dia" y "dia de la semana"
directamente de `ds` para sus terminos de Fourier diarios y semanales, asi que
el componente estacional que ajusta queda desplazado respecto al reloj de
pared local en tantas horas como el huso horario. Eso **no** degrada la
precision del pronostico: el mismo desplazamiento se aplica al ajustar y al
predecir, y se cancela. Solo importaria si se quisiera **interpretar** la
descomposicion de Prophet como "la demanda sube a las 8 de la manana hora
local" para una pagina de explicabilidad —en ese caso, y solo en ese caso,
habria que convertir a `spec.tz_display` antes de ajustar, como hace
`data/calendar.py` para las features de calendario.

Estacionalidad anual sobre historiales cortos
-----------------------------------------------
Con menos de un año de historia, el termino anual no tiene ningun ciclo
completo que ajustar: Prophet lo estima igual, pero su unica funcion es
absorber una parte de la tendencia de baja frecuencia bajo el nombre de
"anual", no una estacionalidad real. Se deja activado porque el enunciado lo
pide y porque no falla ni distorsiona el resto de componentes, pero conviene
no interpretar ese termino como una estacionalidad anual verificada mientras
el dataset de evaluacion cubra menos de un año.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from chronolab.errors import MissingFutrExog
from chronolab.models.protocols import QUANTILES, ModelRequirements, quantile_column
from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId

if TYPE_CHECKING:  # pragma: no cover
    from prophet import Prophet

__all__ = ["ProphetForecaster"]

_PROPHET_ID = ModelId("prophet")
_DEFAULT_REGRESSORS: tuple[str, ...] = ("temp_c",)

# Dos semanas: lo minimo para que la estacionalidad semanal tenga algo que
# ajustar. No es una exigencia de Prophet —fitaria con menos— sino un piso de
# sentido estadistico: por debajo, "semanal" seria una sola observacion del
# patron, indistinguible del ruido.
_MIN_TRAIN_HOURS = 24 * 14


def _require_prophet() -> Any:
    """Importa Prophet bajo demanda, con un mensaje util si falta el extra.

    Returns
    -------
    Any
        La clase `prophet.Prophet`. El tipo de retorno es `Any` a proposito:
        `prophet` esta en la cuarentena de tipos de D16 (docs/ARCHITECTURE.md),
        y fingir un tipo mas preciso aqui no anadiria seguridad real.

    Raises
    ------
    ImportError
        Si `prophet` no esta instalado. El mensaje dice exactamente que
        comando ejecutar.
    """
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra ml
        raise ImportError(
            "chronolab.models.adapters.prophet necesita el extra 'ml': `uv sync --extra ml`."
        ) from exc
    return Prophet


@dataclass(frozen=True, slots=True)
class _FittedProphet:
    """Ajuste de `ProphetForecaster`: un `Prophet` por serie."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    regressors: tuple[str, ...]
    seed: int
    models: dict[str, Prophet]

    @property
    def n_params(self) -> int | None:
        """``None``: Prophet no expone un numero de parametros comparable entre series."""
        return None

    def _horizon(self, futr: FutrFrame | None) -> dict[str, pd.DataFrame]:
        """Trama futura por serie: ``ds`` (y los regresores, si los hay) a partir del `FutrFrame`.

        Parameters
        ----------
        futr
            Exogenas futuras de la ventana. Obligatorio si `self.regressors`
            no esta vacio (`requires.needs_futr_exog` lo exige entonces).

        Returns
        -------
        dict
            ``unique_id -> DataFrame`` con columnas ``ds`` y cada regresor,
            listo para pasarselo a ``Prophet.predict``.

        Raises
        ------
        ChronolabError
            Si `futr` no trae alguno de los regresores registrados en el
            ajuste.
        """
        if futr is None or futr.df.empty:
            grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:]
            base = pd.DataFrame({"ds": grid})
            return dict.fromkeys(self.models, base)

        missing = set(self.regressors) - set(futr.df.columns)
        if missing:
            raise ValueError(
                f"{self.model_id}: faltan los regresores {sorted(missing)} en las "
                "exogenas futuras recibidas"
            )
        columns = ["ds", *self.regressors]
        return {
            str(uid): group.sort_values("ds")[columns].reset_index(drop=True)
            for uid, group in futr.df.groupby("unique_id", sort=False)
        }

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        """Predice `h` pasos para cada serie ajustada.

        Parameters
        ----------
        futr
            Exogenas futuras, con el `ds` a predecir y cada regresor de
            `self.regressors`. Obligatorio si `self.regressors` no esta vacio.
        quantiles
            Cuantiles a estimar, calculados como percentiles empiricos de
            `Prophet.predictive_samples`.

        Returns
        -------
        pandas.DataFrame
            ``unique_id``, ``ds``, ``y_hat`` y una columna por cuantil.

        Raises
        ------
        MissingFutrExog
            Si `self.regressors` no esta vacio y `futr` es ``None``.
        """
        if self.regressors and futr is None:
            raise MissingFutrExog(
                f"{self.model_id} necesita exogenas futuras para sus regresores {self.regressors}"
            )

        future_by_id = self._horizon(futr)
        parts: list[pd.DataFrame] = []
        for uid, model in self.models.items():
            future = future_by_id[uid]
            point = model.predict(future)["yhat"].to_numpy()

            np.random.seed(self.seed)
            samples = model.predictive_samples(future)["yhat"]

            row = pd.DataFrame({"unique_id": uid, "ds": future["ds"].to_numpy(), "y_hat": point})
            for quantile in quantiles:
                row[quantile_column(quantile)] = np.quantile(samples, quantile, axis=1)
            parts.append(row)

        return pd.concat(parts, ignore_index=True)


@dataclass(frozen=True)
class ProphetForecaster:
    """Prophet con estacionalidad diaria/semanal/anual, festivos y temperatura como regresor.

    Un `prophet.Prophet()` por serie (coste O(n_series) por ventana, riesgo
    R2). Cada regresor de `regressors` se registra con
    `Prophet.add_regressor` durante el ajuste, y su valor futuro se lee del
    `FutrFrame` en `predict`.

    Parameters
    ----------
    country_holidays
        Codigo de pais para `Prophet.add_country_holidays` (via el paquete
        `holidays`, ya dependencia del nucleo). ``"ES"`` por defecto: festivos
        de España, coherentes con la serie de demanda electrica de REE que es
        el foco del proyecto. ``None`` desactiva los festivos.
    regressors
        Columnas `futr_exog` a usar como regresores adicionales. ``("temp_c",)``
        por defecto. Tupla vacia para un Prophet puramente univariado —en ese
        caso `requires.needs_futr_exog` es `False`.
    daily_seasonality, weekly_seasonality, yearly_seasonality
        Activan cada componente estacional de Prophet. Las tres a `True` por
        defecto, tal como se pide; ver la nota del modulo sobre la
        estacionalidad anual con historiales cortos.
    interval_width
        Ancho del intervalo que usa `Prophet.predict` para sus propias
        columnas ``yhat_lower``/``yhat_upper`` (que este adaptador no expone
        directamente: los cuantiles salen de `predictive_samples`). Se deja
        configurable porque tambien influye en la escala de la incertidumbre
        de tendencia que Prophet muestrea.
    uncertainty_samples
        Muestras de la distribucion predictiva. 200 por defecto: suficientes
        para estimar siete cuantiles con ruido bajo sin pagar el coste de las
        1000 por defecto de Prophet.
    seed
        Semilla para `predictive_samples`. Ver la nota del modulo sobre por
        que no es una garantia tan fuerte como un generador propio.
    model_id
        Identificador del modelo.

    Raises
    ------
    ValueError
        Si `uncertainty_samples` es menor que uno.
    """

    country_holidays: str | None = "ES"
    regressors: tuple[str, ...] = _DEFAULT_REGRESSORS
    daily_seasonality: bool = True
    weekly_seasonality: bool = True
    yearly_seasonality: bool = True
    interval_width: float = 0.8
    uncertainty_samples: int = 200
    seed: int = 0
    model_id: ModelId = _PROPHET_ID

    def __post_init__(self) -> None:
        """Valida el numero de muestras de incertidumbre."""
        if self.uncertainty_samples < 1:
            raise ValueError(f"uncertainty_samples debe ser >= 1: {self.uncertainty_samples}")

    @property
    def requires(self) -> ModelRequirements:
        """Necesita exogenas futuras solo si hay regresores configurados."""
        return ModelRequirements(
            needs_futr_exog=bool(self.regressors),
            min_context=_MIN_TRAIN_HOURS,
            refit_cost="expensive",
        )

    def fit(self, train: Panel, *, h: int) -> _FittedProphet:
        """Ajusta un `Prophet` por serie, con festivos y regresores registrados.

        Raises
        ------
        KeyError
            Si algun regresor de `self.regressors` no existe en el panel.
        """
        # Nombre distinto del tipo `Prophet` importado bajo TYPE_CHECKING: si
        # se llamase igual, la anotacion local de `models` mas abajo
        # resolveria contra esta variable en tiempo de ejecucion en vez de
        # contra el tipo, y mypy se quejaria de que una variable no es un tipo
        # valido.
        prophet_cls = _require_prophet()
        target = train.spec.target
        columns = ["ds", target, *self.regressors]

        started = perf_counter()
        models: dict[str, Prophet] = {}
        for uid, group in train.df.groupby("unique_id", sort=False):
            frame = group.sort_values("ds")[columns].rename(columns={target: "y"})
            model = prophet_cls(
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                interval_width=self.interval_width,
                uncertainty_samples=self.uncertainty_samples,
            )
            if self.country_holidays:
                model.add_country_holidays(country_name=self.country_holidays)
            for regressor in self.regressors:
                model.add_regressor(regressor)
            model.fit(frame)
            models[str(uid)] = model
        fit_seconds = perf_counter() - started

        return _FittedProphet(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=fit_seconds,
            freq=train.spec.freq,
            regressors=self.regressors,
            seed=self.seed,
            models=models,
        )
