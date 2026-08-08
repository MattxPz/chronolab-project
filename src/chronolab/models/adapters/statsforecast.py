"""Adaptador de statsforecast: AutoARIMA, AutoETS, AutoTheta y MSTL.

Los cuatro son **univariados**: ninguno recibe exogenas. La demanda electrica no
necesita xreg para batir al naive estacional, y mezclar exogenas aqui habria
duplicado la responsabilidad que ya tiene `ProphetForecaster` de este mismo
paquete (`adapters/prophet.py`), que es quien usa la temperatura como
regresor. Mantenerlos univariados tambien evita el efecto colateral de
statsforecast por el que cualquier columna extra en la trama de entrenamiento
se trata como exogena automaticamente: `_univariate_frame` proyecta
deliberadamente a solo ``unique_id, ds, y`` antes de llegar a la libreria.

Import perezoso
----------------
`statsforecast` vive en el extra `ml` (pyproject D20), no en el nucleo. El
modulo tiene que poder **importarse** sin ese extra —lo exige
`tests/unit/test_module_tree.py`, que recorre todo el arbol de paquetes en el
entorno por defecto de CI— asi que `import statsforecast` no aparece a nivel de
modulo. Vive dentro de `_require_statsforecast()`, llamada unicamente al
ajustar un modelo; sin el extra instalado, el error de import se convierte en
un mensaje que dice exactamente que instalar.

AutoARIMA es lento: la mitigacion, medida
------------------------------------------
`AutoARIMA` con sus valores por defecto (`approximation=False`, sin tope de
ordenes) tarda **~32 s** en ajustar una sola serie horaria de 720 puntos en
esta maquina. Con `approximation=True` —minimos cuadrados condicionales en
lugar de maxima verosimilitud exacta durante la busqueda de ordenes— y los
topes por defecto de esta clase (``max_p=3, max_q=3, max_P=1, max_Q=1``), la
misma serie tarda **~0.5-0.7 s**: una mejora de 40-60x que es la razon de que
`approximation=True` sea el valor por defecto aqui y no el de la libreria.
`n_jobs` paraleliza el ajuste **entre series** dentro de una misma llamada;
con solo 2-3 series el coste de arrancar procesos supera al ahorro, asi que el
valor por defecto es ``1`` y subirlo solo compensa con paneles de decenas de
series o mas.

Intervalos conformales: donde son posibles y donde no
-------------------------------------------------------
Los cuatro admiten `ConformalIntervals` de forma nativa y **uniforme**
(`prediction_intervals=...` en el constructor de cada modelo de
statsforecast), a los niveles que se pidan. Pero el intervalo queda fijado al
horizonte `h` con el que se construyo `ConformalIntervals`: pedirle a
statsforecast que extrapole un intervalo conformal mas alla de ese horizonte
lanza un error de forma interna (`ValueError` por descuadre de formas al
sumar los cuantiles). Eso importa aqui porque el motor de backtesting puede
reutilizar un ajuste antiguo en ventanas posteriores cuando
``refit_cost="expensive"`` —la politica por defecto de estos cuatro modelos—,
y una ventana reutilizada pide un horizonte mayor que el original en pasos
desde su propio cutoff. La consecuencia practica:

- **Con intervalos** (``use_intervals=True``, el valor por defecto): un run
  que quiera reutilizar el ajuste entre ventanas tiene que fijar
  ``BacktestPlan(refit_every=1)`` explicitamente, para que cada ventana ajuste
  de nuevo con su propio `h`. `predict()` lo comprueba y lanza un error que
  explica exactamente esto en lugar de dejar que statsforecast falle con un
  mensaje interno sobre formas de array.
- **Sin intervalos** (``use_intervals=False``): el ajuste se puede reutilizar
  sin limite —solo se pide un pronostico puntual mas largo, que a estos
  modelos no les cuesta nada extra calcular—, y la politica por defecto
  (`refit_cost="expensive"`, un unico ajuste por run) funciona sin ajustes.

Porque los intervalos fijan el horizonte de construccion, `h` es tambien
parametro del **constructor** de estas cuatro clases, no solo de `fit`: hace
falta para construir `ConformalIntervals(h=h, ...)` antes de que exista
ningun `Panel` de entrenamiento, y `requires.min_context` —que el motor
consulta antes de la primera ventana— tambien depende de el.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import pandas as pd

from chronolab.models.protocols import QUANTILES, ModelRequirements, quantile_column
from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId

if TYPE_CHECKING:  # pragma: no cover
    from statsforecast import StatsForecast

__all__ = [
    "AutoARIMAForecaster",
    "AutoETSForecaster",
    "AutoThetaForecaster",
    "MSTLForecaster",
]

_AUTO_ARIMA_ID = ModelId("auto_arima")
_AUTO_ETS_ID = ModelId("auto_ets")
_AUTO_THETA_ID = ModelId("auto_theta")
_MSTL_ID = ModelId("mstl")

_DEFAULT_LEVELS: tuple[int, ...] = (80, 95)
_DEFAULT_MSTL_PERIODS: tuple[int, ...] = (24, 168)


def _require_statsforecast() -> tuple[type, type, type, type, type, type]:
    """Importa statsforecast bajo demanda, con un mensaje util si falta el extra.

    Returns
    -------
    tuple
        ``(StatsForecast, AutoARIMA, AutoETS, AutoTheta, MSTL, ConformalIntervals)``
        de la libreria instalada.

    Raises
    ------
    ImportError
        Si `statsforecast` no esta instalado. El mensaje dice exactamente que
        comando ejecutar, en lugar de dejar que el `ModuleNotFoundError` crudo
        de Python llegue a quien esta ejecutando un backtest.
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import MSTL, AutoARIMA, AutoETS, AutoTheta
        from statsforecast.utils import ConformalIntervals
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra ml
        raise ImportError(
            "chronolab.models.adapters.statsforecast necesita el extra 'ml': `uv sync --extra ml`."
        ) from exc
    return StatsForecast, AutoARIMA, AutoETS, AutoTheta, MSTL, ConformalIntervals


def _validate_levels(levels: tuple[int, ...]) -> None:
    """Comprueba que una rejilla de niveles de intervalo es valida.

    Parameters
    ----------
    levels
        Niveles porcentuales, por ejemplo ``(80, 95)``.

    Raises
    ------
    ValueError
        Si algun nivel cae fuera de ``(0, 100)``.
    """
    for level in levels:
        if not 0 < level < 100:
            raise ValueError(f"nivel de intervalo fuera de (0, 100): {level}")


def _validate_calibration_windows(windows: int) -> None:
    """Comprueba que el numero de ventanas de calibracion conformal es admisible.

    Parameters
    ----------
    windows
        Valor de `calibration_windows`.

    Raises
    ------
    ValueError
        Si es menor que dos: `ConformalIntervals` de statsforecast exige al
        menos dos ventanas internas para poder calibrar el intervalo, y con
        una sola no hay residuos suficientes para estimar su dispersion.
    """
    if windows < 2:
        raise ValueError(f"calibration_windows debe ser >= 2: {windows}")


def _interval_quantiles(level: int) -> tuple[float, float]:
    """Cuantiles inferior y superior de un intervalo central de nivel `level`.

    Parameters
    ----------
    level
        Nivel porcentual, por ejemplo ``95``.

    Returns
    -------
    tuple
        ``((100 - level) / 200, 1 - (100 - level) / 200)``, por ejemplo
        ``(0.025, 0.975)`` para ``level=95``. Coincide exactamente con dos
        cuantiles de la rejilla canonica del proyecto para ``80`` y ``95``.
    """
    lower = (100 - level) / 200.0
    return lower, 1.0 - lower


def _steps_between(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> int:
    """Pasos de rejilla entre dos marcas de tiempo, a una frecuencia dada.

    Parameters
    ----------
    start, end
        Extremos, con ``end >= start``.
    freq
        Alias de offset de pandas.

    Returns
    -------
    int
        Numero de pasos de `freq` entre `start` y `end`, ambos inclusive en la
        rejilla generada (``0`` si coinciden).
    """
    return len(pd.date_range(start, end, freq=freq)) - 1


def _assert_matching_horizon(model_id: ModelId, configured_h: int, requested_h: int) -> None:
    """Comprueba que el `h` del plan coincide con el que se fijo al construir el modelo.

    Parameters
    ----------
    model_id
        Modelo, para el mensaje.
    configured_h
        Horizonte fijado en el constructor de la clase `Forecaster`.
    requested_h
        Horizonte que pide `BacktestPlan` en esta llamada a `fit`.

    Raises
    ------
    ValueError
        Si no coinciden. `ConformalIntervals` fija el horizonte de calibracion
        en la construccion del modelo, antes de que exista ningun `Panel`; si
        el plan pide otro `h`, el intervalo calibrado no significaria lo que
        dice significar.
    """
    if configured_h != requested_h:
        raise ValueError(
            f"{model_id}: se construyo con h={configured_h} pero el plan de "
            f"backtesting pide h={requested_h}. Los intervalos conformales fijan "
            f"el horizonte al construir el modelo: crea una instancia con "
            f"h={requested_h}."
        )


def _univariate_frame(panel: Panel) -> pd.DataFrame:
    """Proyecta un panel a la trama minima que espera statsforecast: unique_id, ds, y.

    Parameters
    ----------
    panel
        Panel de entrenamiento.

    Returns
    -------
    pandas.DataFrame
        Sin ninguna columna de exogena. Es la barrera contra el efecto
        colateral de statsforecast por el que cualquier columna extra en la
        trama de ajuste se interpreta como regresor: aqui, ausencia fisica.
    """
    target = panel.spec.target
    frame = panel.df[["unique_id", "ds", target]]
    if target != "y":
        frame = frame.rename(columns={target: "y"})
    return _impute_target(frame.reset_index(drop=True))


def _impute_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Rellena los huecos de la objetivo antes de entregarla a statsforecast.

    Los cuatro modelos de este modulo declaran ``handles_nan_target=False``, y
    el contrato de `Forecaster.fit` (docs/ARCHITECTURE.md §5.2) dice que en ese
    caso **el adaptador imputa dentro de `fit`**. Sin esto, un panel con un
    hueco —que el invariante I3 conserva como `NaN` explicito, no como fila
    ausente— hace que `mstl` aborte con "cannot handle missing values" y que el
    motor registre la ventana como fallida. El resultado no seria un modelo
    peor sino un tramo entero sin predicciones, y con el sin residuos que
    puntuar.

    El relleno es hacia delante: mira solo al pasado, que es la unica
    imputacion admisible del proyecto. Un hueco al principio del tramo no tiene
    pasado del que tirar y se rellena con la media del propio tramo de
    entrenamiento; como el motor ya recorto a ``ds <= cutoff``, ese estadistico
    tampoco puede haber visto el futuro.

    Parameters
    ----------
    frame
        Trama ``unique_id``, ``ds``, ``y`` del tramo de entrenamiento.

    Returns
    -------
    pandas.DataFrame
        La misma trama con `y` sin huecos. Si no habia ninguno se devuelve tal
        cual, sin copiar.
    """
    if not frame["y"].isna().any():
        return frame
    filled = frame.copy()
    filled["y"] = filled.groupby("unique_id", sort=False)["y"].ffill()
    means = filled.groupby("unique_id", sort=False)["y"].transform("mean")
    # El `0.0` final cubre una serie sin ninguna observacion en el tramo: su
    # media es `NaN` y statsforecast seguiria abortando.
    filled["y"] = filled["y"].fillna(means).fillna(0.0)
    return filled


def _assign_quantiles(
    frame: pd.DataFrame,
    *,
    alias: str,
    levels: tuple[int, ...],
    quantiles: Sequence[float],
) -> pd.DataFrame:
    """Traduce las columnas ``<alias>``/``<alias>-lo-L``/``<alias>-hi-L`` a la rejilla canonica.

    Parameters
    ----------
    frame
        Trama con la salida cruda de `StatsForecast.predict`, para una sola
        serie y ya recortada al tramo pedido.
    alias
        Nombre de columna del pronostico puntual en `frame`.
    levels
        Niveles de intervalo presentes en `frame`. Tupla vacia si el modelo se
        ajusto sin `ConformalIntervals`.
    quantiles
        Cuantiles pedidos por el motor.

    Returns
    -------
    pandas.DataFrame
        ``unique_id``, ``ds``, ``y_hat`` y una columna por cada cuantil pedido
        que se pueda derivar de `levels` o que sea ``0.5`` (el pronostico
        puntual). Los cuantiles pedidos que no correspondan a ningun nivel
        presente simplemente no aparecen: el motor los rellena con `NaN`,
        que es preferible a inventar un intervalo con un nivel no calibrado.
    """
    result = frame[["unique_id", "ds"]].copy()
    result["y_hat"] = frame[alias]

    for quantile in quantiles:
        column = quantile_column(quantile)
        if math.isclose(quantile, 0.5, abs_tol=1e-9):
            result[column] = frame[alias]
            continue
        for level in levels:
            lower, upper = _interval_quantiles(level)
            if math.isclose(quantile, lower, abs_tol=1e-9):
                result[column] = frame[f"{alias}-lo-{level}"]
                break
            if math.isclose(quantile, upper, abs_tol=1e-9):
                result[column] = frame[f"{alias}-hi-{level}"]
                break

    return result


@dataclass(frozen=True, slots=True)
class _FittedStatsForecastModel:
    """Ajuste comun a los cuatro adaptadores: un `StatsForecast` ya ajustado."""

    model_id: ModelId
    alias: str
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    levels: tuple[int, ...]
    sf: StatsForecast
    series_ids: tuple[str, ...]

    @property
    def n_params(self) -> int | None:
        """``None``: statsforecast no expone un numero de parametros uniforme entre modelos."""
        return None

    def _horizon(self, futr: FutrFrame | None) -> dict[str, list[pd.Timestamp]]:
        """Instantes a predecir por serie: los del `FutrFrame` si lo hay, o `cutoff + h*freq`."""
        if futr is not None and not futr.df.empty:
            return {
                str(uid): group.sort_values("ds")["ds"].tolist()
                for uid, group in futr.df.groupby("unique_id", sort=False)
            }
        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:].tolist()
        return dict.fromkeys(self.series_ids, grid)

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        """Predice para todas las series del ajuste, en los instantes pedidos.

        Parameters
        ----------
        futr
            Ignorado como fuente de exogenas —estos modelos no las usan—, pero
            consultado para el `ds` exacto cuando esta disponible, igual que en
            `chronolab.models.baselines`.
        quantiles
            Cuantiles a estimar. Los que no correspondan a ningun nivel
            calibrado, ni sean ``0.5``, quedan fuera de la trama devuelta.

        Returns
        -------
        pandas.DataFrame
            ``unique_id``, ``ds``, ``y_hat`` y las columnas de cuantil que se
            puedan derivar.

        Raises
        ------
        ValueError
            Si el horizonte pedido —en pasos desde `cutoff`— supera `h` y el
            ajuste tiene intervalos conformales: statsforecast no puede
            extrapolarlos. Ver el docstring del modulo.
        """
        horizon = self._horizon(futr)
        target_max = max(ds for instants in horizon.values() for ds in instants)
        steps_needed = _steps_between(self.cutoff, target_max, self.freq)

        if steps_needed < 1:
            raise ValueError(f"{self.model_id}: nada que predecir despues de {self.cutoff}")
        if self.levels and steps_needed > self.h:
            raise ValueError(
                f"{self.model_id}: hace falta un horizonte de {steps_needed} pasos desde "
                f"el ajuste ({self.cutoff}), pero los intervalos conformales solo estan "
                f"calibrados para h={self.h}. Statsforecast no puede extrapolar un "
                "intervalo conformal mas alla de su horizonte de calibracion: fija "
                "refit_every=1 en el BacktestPlan para que cada ventana reajuste con su "
                "propio h, o construye el modelo con use_intervals=False si necesitas "
                "reutilizar el ajuste sin cuantiles."
            )

        raw = self.sf.predict(
            h=steps_needed, level=list(self.levels) if self.levels else None
        ).set_index(["unique_id", "ds"])

        parts: list[pd.DataFrame] = []
        for uid, instants in horizon.items():
            selected = raw.loc[[(uid, ds) for ds in instants]].reset_index()
            parts.append(
                _assign_quantiles(
                    selected, alias=self.alias, levels=self.levels, quantiles=quantiles
                )
            )
        return pd.concat(parts, ignore_index=True)


def _fit_with_statsforecast(
    train: Panel,
    h: int,
    *,
    sf_model: object,
    alias: str,
    model_id: ModelId,
    levels: tuple[int, ...],
    n_jobs: int,
) -> _FittedStatsForecastModel:
    """Rutina de ajuste comun: construye un `StatsForecast` de un solo modelo y lo ajusta.

    Parameters
    ----------
    train
        Panel de entrenamiento, ya recortado por el motor a ``ds <= cutoff``.
    h
        Horizonte solicitado por el plan.
    sf_model
        Instancia ya construida de un modelo de `statsforecast.models`.
    alias
        Nombre de columna que producira ese modelo en `predict`.
    model_id
        Identificador de `chronolab` para el `Forecaster`.
    levels
        Niveles de intervalo con los que se construyo `sf_model`, o tupla
        vacia si no lleva `prediction_intervals`.
    n_jobs
        Procesos para paralelizar el ajuste entre series.

    Returns
    -------
    _FittedStatsForecastModel
        Con el tiempo de ajuste medido con `perf_counter`.
    """
    StatsForecast, *_ = _require_statsforecast()
    started = perf_counter()
    sf = StatsForecast(models=[sf_model], freq=train.spec.freq, n_jobs=n_jobs)
    sf.fit(_univariate_frame(train))
    fit_seconds = perf_counter() - started
    return _FittedStatsForecastModel(
        model_id=model_id,
        alias=alias,
        cutoff=train.last_ds,
        h=h,
        fit_seconds=fit_seconds,
        freq=train.spec.freq,
        levels=levels,
        sf=sf,
        series_ids=train.ids(),
    )


@dataclass(frozen=True)
class AutoARIMAForecaster:
    """ARIMA estacional con seleccion automatica de ordenes, via `statsforecast.AutoARIMA`.

    Parameters
    ----------
    h
        Horizonte del plan de backtesting. Necesario en el constructor porque
        `ConformalIntervals` fija el horizonte de calibracion al construirse,
        antes de que exista ningun `Panel` (ver docstring del modulo).
    season_length
        Longitud estacional en pasos. 24 por defecto: el ciclo diario de una
        serie horaria.
    max_p, max_q, max_P, max_Q
        Topes de orden de la busqueda `stepwise`. Los valores por defecto
        (``3, 3, 1, 1``) son deliberadamente modestos: con ellos y
        `approximation=True`, ajustar una serie de 720 puntos tarda medio
        segundo en lugar de treinta.
    approximation
        Usar minimos cuadrados condicionales en vez de maxima verosimilitud
        exacta durante la busqueda de ordenes. Es la mitigacion medida contra
        la lentitud de `AutoARIMA` sobre series horarias largas (ver
        docstring del modulo); desactivarla para un ajuste final de mas
        calidad cuando el tiempo ya no sea la restriccion.
    use_intervals
        Ajustar con `ConformalIntervals` a `levels`. Si es `False`, el ajuste
        se puede reutilizar entre ventanas sin la limitacion de horizonte que
        describe el docstring del modulo, pero `predict` no produce cuantiles.
    levels
        Niveles de intervalo, en `(0, 100)`.
    calibration_windows
        Ventanas internas que usa statsforecast para calibrar el intervalo
        conformal (parametro `n_windows` de `ConformalIntervals`; no confundir
        con las ventanas del backtest de chronolab). Statsforecast exige al
        menos ``calibration_windows * h + 1`` observaciones para calibrar.
    n_jobs
        Procesos para paralelizar el ajuste entre series de un mismo panel.
        Con 2-3 series el coste de arrancar procesos supera al ahorro; solo
        compensa con paneles de decenas de series o mas.
    model_id
        Identificador del modelo.

    Raises
    ------
    ValueError
        Si algun nivel de `levels` cae fuera de ``(0, 100)`` o si
        `calibration_windows` es menor que uno.
    """

    h: int
    season_length: int = 24
    max_p: int = 3
    max_q: int = 3
    max_P: int = 1
    max_Q: int = 1
    approximation: bool = True
    use_intervals: bool = True
    levels: tuple[int, ...] = _DEFAULT_LEVELS
    calibration_windows: int = 2
    n_jobs: int = 1
    model_id: ModelId = _AUTO_ARIMA_ID

    def __post_init__(self) -> None:
        """Valida niveles y ventanas de calibracion."""
        _validate_levels(self.levels)
        _validate_calibration_windows(self.calibration_windows)

    @property
    def requires(self) -> ModelRequirements:
        """Entrenamiento minimo: dos estaciones completas, y lo que exija la calibracion conformal."""
        base = 2 * self.season_length
        min_context = (
            max(base, self.calibration_windows * self.h + 1) if self.use_intervals else base
        )
        return ModelRequirements(min_context=min_context, refit_cost="expensive")

    def fit(self, train: Panel, *, h: int) -> _FittedStatsForecastModel:
        """Ajusta un `AutoARIMA` sobre cada serie del panel de entrenamiento."""
        _assert_matching_horizon(self.model_id, self.h, h)
        _, SFAutoARIMA, _, _, _, ConformalIntervals = _require_statsforecast()
        intervals = (
            ConformalIntervals(h=self.h, n_windows=self.calibration_windows)
            if self.use_intervals
            else None
        )
        model = SFAutoARIMA(
            season_length=self.season_length,
            max_p=self.max_p,
            max_q=self.max_q,
            max_P=self.max_P,
            max_Q=self.max_Q,
            approximation=self.approximation,
            prediction_intervals=intervals,
            alias="AutoARIMA",
        )
        return _fit_with_statsforecast(
            train,
            h,
            sf_model=model,
            alias="AutoARIMA",
            model_id=self.model_id,
            levels=self.levels if self.use_intervals else (),
            n_jobs=self.n_jobs,
        )


@dataclass(frozen=True)
class AutoETSForecaster:
    """Suavizado exponencial con seleccion automatica de componentes, via `statsforecast.AutoETS`.

    Parameters
    ----------
    h
        Ver `AutoARIMAForecaster.h`.
    season_length
        Longitud estacional en pasos.
    model
        Cadena de tres letras que fija (o deja automatica con ``"Z"``) el tipo
        de error, tendencia y estacionalidad, en la notacion de Hyndman.
        ``"ZZZ"`` deja las tres en busqueda automatica.
    use_intervals, levels, calibration_windows, n_jobs, model_id
        Ver `AutoARIMAForecaster`.

    Raises
    ------
    ValueError
        Si algun nivel de `levels` cae fuera de ``(0, 100)`` o si
        `calibration_windows` es menor que uno.
    """

    h: int
    season_length: int = 24
    model: str = "ZZZ"
    use_intervals: bool = True
    levels: tuple[int, ...] = _DEFAULT_LEVELS
    calibration_windows: int = 2
    n_jobs: int = 1
    model_id: ModelId = _AUTO_ETS_ID

    def __post_init__(self) -> None:
        """Valida niveles y ventanas de calibracion."""
        _validate_levels(self.levels)
        _validate_calibration_windows(self.calibration_windows)

    @property
    def requires(self) -> ModelRequirements:
        """Entrenamiento minimo: dos estaciones completas, y lo que exija la calibracion conformal."""
        base = 2 * self.season_length
        min_context = (
            max(base, self.calibration_windows * self.h + 1) if self.use_intervals else base
        )
        return ModelRequirements(min_context=min_context, refit_cost="expensive")

    def fit(self, train: Panel, *, h: int) -> _FittedStatsForecastModel:
        """Ajusta un `AutoETS` sobre cada serie del panel de entrenamiento."""
        _assert_matching_horizon(self.model_id, self.h, h)
        _, _, SFAutoETS, _, _, ConformalIntervals = _require_statsforecast()
        intervals = (
            ConformalIntervals(h=self.h, n_windows=self.calibration_windows)
            if self.use_intervals
            else None
        )
        model = SFAutoETS(
            season_length=self.season_length,
            model=self.model,
            prediction_intervals=intervals,
            alias="AutoETS",
        )
        return _fit_with_statsforecast(
            train,
            h,
            sf_model=model,
            alias="AutoETS",
            model_id=self.model_id,
            levels=self.levels if self.use_intervals else (),
            n_jobs=self.n_jobs,
        )


@dataclass(frozen=True)
class AutoThetaForecaster:
    """Metodo Theta con seleccion automatica de variante, via `statsforecast.AutoTheta`.

    Parameters
    ----------
    h
        Ver `AutoARIMAForecaster.h`.
    season_length
        Longitud estacional en pasos.
    decomposition_type
        ``"multiplicative"`` o ``"additive"``. Multiplicativa por defecto: la
        demanda electrica tipica tiene amplitud estacional proporcional al
        nivel, no constante.
    use_intervals, levels, calibration_windows, n_jobs, model_id
        Ver `AutoARIMAForecaster`.

    Raises
    ------
    ValueError
        Si algun nivel de `levels` cae fuera de ``(0, 100)`` o si
        `calibration_windows` es menor que uno.
    """

    h: int
    season_length: int = 24
    decomposition_type: str = "multiplicative"
    use_intervals: bool = True
    levels: tuple[int, ...] = _DEFAULT_LEVELS
    calibration_windows: int = 2
    n_jobs: int = 1
    model_id: ModelId = _AUTO_THETA_ID

    def __post_init__(self) -> None:
        """Valida niveles y ventanas de calibracion."""
        _validate_levels(self.levels)
        _validate_calibration_windows(self.calibration_windows)

    @property
    def requires(self) -> ModelRequirements:
        """Entrenamiento minimo: dos estaciones completas, y lo que exija la calibracion conformal."""
        base = 2 * self.season_length
        min_context = (
            max(base, self.calibration_windows * self.h + 1) if self.use_intervals else base
        )
        return ModelRequirements(min_context=min_context, refit_cost="expensive")

    def fit(self, train: Panel, *, h: int) -> _FittedStatsForecastModel:
        """Ajusta un `AutoTheta` sobre cada serie del panel de entrenamiento."""
        _assert_matching_horizon(self.model_id, self.h, h)
        _, _, _, SFAutoTheta, _, ConformalIntervals = _require_statsforecast()
        intervals = (
            ConformalIntervals(h=self.h, n_windows=self.calibration_windows)
            if self.use_intervals
            else None
        )
        model = SFAutoTheta(
            season_length=self.season_length,
            decomposition_type=self.decomposition_type,
            prediction_intervals=intervals,
            alias="AutoTheta",
        )
        return _fit_with_statsforecast(
            train,
            h,
            sf_model=model,
            alias="AutoTheta",
            model_id=self.model_id,
            levels=self.levels if self.use_intervals else (),
            n_jobs=self.n_jobs,
        )


@dataclass(frozen=True)
class MSTLForecaster:
    """Descomposicion STL multiple con `AutoARIMA` como pronosticador de tendencia.

    Descompone la serie en varios componentes estacionales via STL —uno por
    periodo de `season_lengths`— y pronostica la tendencia residual con
    `AutoARIMA`; los componentes estacionales se extrapolan con el ultimo
    ciclo observado. Con periodos ``[24, 168]`` (diario y semanal) suele ser
    el ganador estadistico sobre series horarias con doble estacionalidad,
    precisamente porque es el unico de los cuatro adaptadores de este modulo
    que modela **las dos** en lugar de una sola.

    Parameters
    ----------
    h
        Ver `AutoARIMAForecaster.h`.
    season_lengths
        Periodos estacionales en pasos, uno por componente STL. ``(24, 168)``
        por defecto: ciclo diario y semanal de una serie horaria.
    trend_max_p, trend_max_q
        Topes de orden del `AutoARIMA` de tendencia. Mas bajos que los de
        `AutoARIMAForecaster` a proposito: la tendencia ya viene sin
        estacionalidad (la STL se la ha quitado), asi que necesita menos
        terminos para explicarla.
    approximation
        Ver `AutoARIMAForecaster.approximation`; se aplica al `AutoARIMA` de
        tendencia interno.
    use_intervals, levels, calibration_windows, n_jobs, model_id
        Ver `AutoARIMAForecaster`.

    Raises
    ------
    ValueError
        Si `season_lengths` esta vacia, si algun periodo es menor que dos, si
        algun nivel de `levels` cae fuera de ``(0, 100)``, o si
        `calibration_windows` es menor que uno.
    """

    h: int
    season_lengths: tuple[int, ...] = _DEFAULT_MSTL_PERIODS
    trend_max_p: int = 2
    trend_max_q: int = 2
    approximation: bool = True
    use_intervals: bool = True
    levels: tuple[int, ...] = _DEFAULT_LEVELS
    calibration_windows: int = 2
    n_jobs: int = 1
    model_id: ModelId = _MSTL_ID

    def __post_init__(self) -> None:
        """Valida periodos, niveles y ventanas de calibracion."""
        if not self.season_lengths:
            raise ValueError("season_lengths no puede estar vacia")
        if any(period < 2 for period in self.season_lengths):
            raise ValueError(f"cada periodo debe ser >= 2: {self.season_lengths}")
        _validate_levels(self.levels)
        _validate_calibration_windows(self.calibration_windows)

    @property
    def requires(self) -> ModelRequirements:
        """Entrenamiento minimo: dos ciclos del periodo mas largo, y lo que exija la calibracion."""
        base = 2 * max(self.season_lengths)
        min_context = (
            max(base, self.calibration_windows * self.h + 1) if self.use_intervals else base
        )
        return ModelRequirements(min_context=min_context, refit_cost="expensive")

    def fit(self, train: Panel, *, h: int) -> _FittedStatsForecastModel:
        """Ajusta un `MSTL` con `AutoARIMA` de tendencia sobre cada serie."""
        _assert_matching_horizon(self.model_id, self.h, h)
        _, SFAutoARIMA, _, _, SFMSTL, ConformalIntervals = _require_statsforecast()
        intervals = (
            ConformalIntervals(h=self.h, n_windows=self.calibration_windows)
            if self.use_intervals
            else None
        )
        trend_forecaster = SFAutoARIMA(
            max_p=self.trend_max_p,
            max_q=self.trend_max_q,
            approximation=self.approximation,
        )
        model = SFMSTL(
            season_length=list(self.season_lengths),
            trend_forecaster=trend_forecaster,
            prediction_intervals=intervals,
            alias="MSTL",
        )
        return _fit_with_statsforecast(
            train,
            h,
            sf_model=model,
            alias="MSTL",
            model_id=self.model_id,
            levels=self.levels if self.use_intervals else (),
            n_jobs=self.n_jobs,
        )
