"""Motor de backtesting: recorre ventanas por modelo y persiste artefactos.

Aplica la politica de refit, verifica el cutoff en cada prediccion y registra
los fallos con `status` en lugar de silenciarlos.

Las cuatro garantias anti-fuga del motor son **estructurales**, no de
convencion. Ninguna depende de que quien escriba un adaptador se acuerde de
nada:

1. **El modelo nunca ve el panel completo.** Lo unico que cruza hacia `fit` es
   ``panel.train(window)``, recortado a ``ds <= cutoff``. Como `Panel` no expone
   `scale`, `impute` ni `transform`, y el proyecto no tiene etapa global de
   preprocesado, cualquier estadistico de escalado o normalizacion solo puede
   haberse ajustado con la particion de entrenamiento de la ventana en curso.
   El motor lo comprueba antes de entregar el panel (`_assert_train_at_cutoff`).
2. **Las exogenas historicas se cortan en el cutoff.** Lo que se pasa al tramo de
   prediccion es un `FutrFrame`, que fisicamente solo contiene columnas
   `futr_exog`. El motor verifica la trama que recibe del proveedor: si trae la
   objetivo o una `hist_exog`, el run se detiene con `LeakageError`.
3. **Nada de lo que se predice cae en el pasado conocido.** Toda `ds` de la
   prediccion se compara con el cutoff de la ventana, en el unico camino que
   produce predicciones.
4. **El objeto ajustado no puede adelantarse a su ventana.** Al reutilizar un
   ajuste por politica de refit se comprueba ``fitted.cutoff <= window.cutoff``.
   Reutilizar es obsolescencia —legitima y registrada—; lo contrario seria fuga.

Como sabe un modelo a que instantes predice: el `FutrFrame` lleva las marcas de
tiempo del tramo evaluado, y las lleva **aunque el panel no declare ninguna
exogena futura**, en cuyo caso solo trae `unique_id` y `ds`. Ese es el canal. Un
modelo sin ese dato solo puede predecir desde su propio cutoff, de modo que con
``gap > 0``, o al reutilizar un ajuste de una ventana anterior, sus marcas no
coincidiran con las evaluadas y el motor lo dira: contrato incumplido si predice
otros instantes del futuro, `CutoffViolation` si predice instantes ya conocidos.
Alinear en silencio lo que devuelva seria convertir un off-by-one en un resultado.

Alcance: este modulo produce los artefactos **en memoria**. El esquema de las
tramas es el de docs/ARCHITECTURE.md §7.4, sin la columna `run_id`, que la anade
`chronolab.artifacts.writer` al persistir porque alli es clave de ruta.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from chronolab.errors import (
    ChronolabError,
    CutoffViolation,
    LeakageError,
    MissingFutrExog,
    PredictionContractError,
)
from chronolab.evaluation.splitters import RollingOriginSplitter, Window
from chronolab.models.protocols import QUANTILES, FittedForecaster, Forecaster, quantile_column
from chronolab.panel import FutrFrame, Panel, PanelSpec
from chronolab.types import ModelId, RefitCost, SplitMode, Vintage

if TYPE_CHECKING:  # pragma: no cover
    # Import diferido: en tiempo de ejecucion el motor solo llama a `.futr(...)`,
    # asi que `evaluation` no depende de `data`. El protocolo es estructural.
    from chronolab.data.futr import FutrProvider

__all__ = ["FORECAST_KEY_COLUMNS", "BacktestPlan", "BacktestResult", "backtest"]

FORECAST_KEY_COLUMNS: tuple[str, ...] = (
    "unique_id",
    "model_id",
    "window_id",
    "cutoff",
    "ds",
    "h_step",
    "y",
    "y_hat",
)
"""Columnas de `forecasts` anteriores a las de cuantil (docs/ARCHITECTURE.md §7.4).

`model_id` es clave de particion en disco y columna en memoria; `run_id` lo anade
el escritor de artefactos. `y` viaja desnormalizada a proposito: es el eje de casi
todos los graficos y la app no puede unir tablas grandes en caliente.
"""

_FAILED = "failed"
_OK = "ok"
_SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class BacktestPlan:
    """Plan de un run: como se parte el tiempo y que se le pide a cada modelo.

    Un run cubre un dataset, un plan y un vintage de exogenas. Comparar modelos
    solo es legitimo dentro de un run, y esta restriccion es lo que elimina la
    clase de errores "compare un MASE calculado con otras ventanas".

    Parameters
    ----------
    h
        Horizonte de prediccion en pasos.
    n_windows
        Numero de ventanas del plan.
    step_size
        Separacion entre cutoffs consecutivos, en pasos. Si es menor que `h` los
        tramos evaluados se solapan: se permite, pero afecta a la independencia
        que asume Diebold-Mariano y por eso queda registrado.
    gap
        Pasos descartados entre el cutoff y la primera prediccion. Emula latencia
        de datos y corta la autocorrelacion de corto alcance.
    mode
        ``"expanding"`` (el entrenamiento crece) o ``"sliding"`` (longitud fija).
    train_size
        Longitud del entrenamiento en pasos. Obligatoria en modo deslizante.
    holdout_windows
        Ventanas finales reservadas para reportar. El tuning solo puede mirar las
        de desarrollo.
    min_train_size
        Entrenamiento minimo, en pasos, para que una ventana entre en el plan.
        Es el `min_context` del splitter; se llama distinto aqui para no
        confundirlo con `ModelRequirements.min_context`, que es una exigencia
        **del modelo** y produce ventanas saltadas, no ventanas inexistentes.
    quantiles
        Rejilla de cuantiles que se pide a los modelos.
    refit_every
        Cada cuantas ventanas se reajusta. ``None`` aplica la politica por
        defecto segun `ModelRequirements.refit_cost`: un ajuste por ventana para
        los baratos y uno solo por run para los caros.
    seed
        Semilla global del run.

    Raises
    ------
    WindowValidationError
        Si los parametros de particion son incoherentes.
    ValueError
        Si la rejilla de cuantiles o `refit_every` no son validos.
    """

    h: int
    n_windows: int
    step_size: int = 1
    gap: int = 0
    mode: SplitMode = "expanding"
    train_size: int | None = None
    holdout_windows: int = 0
    min_train_size: int = 1
    quantiles: tuple[float, ...] = QUANTILES
    refit_every: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """Valida el plan completo, incluida la parte que delega en el splitter."""
        if list(self.quantiles) != sorted(set(self.quantiles)):
            raise ValueError(f"quantiles debe ser estrictamente creciente: {self.quantiles}")
        for quantile in self.quantiles:
            quantile_column(quantile)  # valida el rango (0, 1)
        if self.refit_every is not None and self.refit_every < 1:
            raise ValueError(f"refit_every debe ser >= 1: {self.refit_every}")
        self.splitter()

    def splitter(self) -> RollingOriginSplitter:
        """Splitter que materializa la parte temporal del plan.

        Returns
        -------
        RollingOriginSplitter
            Unico emisor de particiones del proyecto.
        """
        return RollingOriginSplitter(
            h=self.h,
            n_windows=self.n_windows,
            step_size=self.step_size,
            gap=self.gap,
            mode=self.mode,
            train_size=self.train_size,
            holdout_windows=self.holdout_windows,
            min_context=self.min_train_size,
        )

    @property
    def quantile_columns(self) -> tuple[str, ...]:
        """Nombres de columna de los cuantiles del plan, en orden creciente."""
        return tuple(quantile_column(q) for q in self.quantiles)

    def refit_every_for(self, cost: RefitCost, n_windows: int) -> int:
        """Politica de refit aplicable a un modelo.

        Parameters
        ----------
        cost
            Coste declarado de reajustar el modelo.
        n_windows
            Ventanas efectivas del run.

        Returns
        -------
        int
            `refit_every` explicito del plan si lo hay; si no, ``1`` para modelos
            ``free`` y ``cheap``, y `n_windows` (un unico ajuste) para los
            ``expensive``. Reutilizar un ajuste no es fuga —el ajuste es siempre
            anterior a la ventana— pero cambia el resultado, asi que se registra.

        Notes
        -----
        Un ajuste reutilizado predice sobre una ventana cuyo cutoff es posterior
        al suyo, asi que el modelo necesita saber a que instantes predecir: sin
        `FutrProvider` en el run, un modelo que predice desde su propio cutoff
        devolvera marcas ya conocidas y el motor detendra el run.
        """
        if self.refit_every is not None:
            return self.refit_every
        return n_windows if cost == "expensive" else 1


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Artefactos en memoria de un run de backtesting.

    Attributes
    ----------
    forecasts
        Tabla de hechos, en formato largo: una fila por
        ``(modelo, serie, ventana, instante)``. Columnas `FORECAST_KEY_COLUMNS`
        mas una por cuantil. Es el artefacto que consume todo lo demas:
        metricas, tests estadisticos, detectores de anomalias y la app.
    windows
        Una fila por ventana efectiva del run.
    model_runs
        Una fila por ``(modelo, ventana)``, incluidas las que fallaron o se
        saltaron: un modelo que revienta en tres ventanas de veinte se ve.
    spec
        Especificacion del panel evaluado.
    plan
        Plan que genero el run.
    futr_vintage
        Vintage de las exogenas futuras, o ``None`` si el run no uso ninguna.
        Comparar filas de vintages distintos no es legitimo, y por eso viaja
        pegado al resultado en vez de reconstruirse despues.
    """

    forecasts: pd.DataFrame
    windows: pd.DataFrame
    model_runs: pd.DataFrame
    spec: PanelSpec
    plan: BacktestPlan
    futr_vintage: Vintage | None


def backtest(
    panel: Panel,
    models: Sequence[Forecaster],
    plan: BacktestPlan,
    *,
    futr: "FutrProvider | None" = None,
) -> BacktestResult:
    """Ejecuta un run de backtesting de origen rodante.

    El orden de operaciones es el de docs/ARCHITECTURE.md §6.2: se parten las
    ventanas una sola vez, y para cada modelo se recorren en orden temporal
    ajustando segun la politica de refit, pidiendo las exogenas futuras de cada
    ventana y prediciendo su tramo de evaluacion.

    Parameters
    ----------
    panel
        Panel canonico completo. El motor lo recorta por ventana; los modelos
        nunca lo reciben entero.
    models
        Modelos a evaluar. Sus `model_id` deben ser distintos: son la clave de
        particion de los artefactos.
    plan
        Plan de backtesting.
    futr
        Proveedor de exogenas futuras. Obligatorio si algun modelo declara
        `needs_futr_exog`. Su `vintage` viaja al resultado.

    Returns
    -------
    BacktestResult
        Artefactos en memoria del run.

    Raises
    ------
    ValueError
        Si no hay modelos o si dos comparten `model_id`.
    MissingFutrExog
        Si un modelo necesita exogenas futuras y el run no puede darselas. El run
        aborta entero en lugar de evaluar ese modelo a ciegas.
    WindowValidationError
        Si el panel no da para el plan.
    LeakageError
        Si en algun punto del bucle se detecta informacion posterior al cutoff.
        Nunca se captura para continuar: un run con fuga no produce resultados
        publicables.
    """
    if not models:
        raise ValueError("un run necesita al menos un modelo")
    model_ids = [model.model_id for model in models]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError(f"model_id repetido entre los modelos del run: {model_ids}")
    _require_futr_provider(models, panel.spec, futr)

    windows = plan.splitter().split(panel)
    forecast_parts: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []

    for model in models:
        refit_every = plan.refit_every_for(model.requires.refit_cost, len(windows))
        fitted: FittedForecaster | None = None
        windows_since_fit = 0

        for window in windows:
            train = panel.train(window)
            _assert_train_at_cutoff(train, window)
            train_steps = int(train.df["ds"].nunique())
            if train_steps < model.requires.min_context:
                records.append(
                    _record(
                        model,
                        window,
                        status=_SKIPPED,
                        error=(
                            f"entrenamiento de {train_steps} pasos < "
                            f"min_context={model.requires.min_context}"
                        ),
                        refit=False,
                        refit_every=refit_every,
                    )
                )
                continue

            # Las exogenas futuras se piden y se verifican **fuera** del bloque
            # que atrapa fallos de modelo: un proveedor roto no es un modelo
            # roto. Tratarlo como tal marcaria como fallidos a todos los modelos
            # del run y el leaderboard senalaria al sitio equivocado.
            futr_frame = None
            if futr is not None:
                futr_frame = futr.futr(window, ids=train.ids())
                _assert_futr_frame(futr_frame, window, panel.spec)

            refit = fitted is None or windows_since_fit >= refit_every
            fit_seconds = 0.0
            predict_seconds = 0.0
            try:
                if refit:
                    started = perf_counter()
                    fitted = model.fit(train, h=window.h)
                    fit_seconds = perf_counter() - started
                    windows_since_fit = 0
                if fitted is None:  # pragma: no cover  refit garantiza el ajuste
                    raise ChronolabError(f"{model.model_id} no ha producido un ajuste")
                _assert_fitted_at_or_before_cutoff(fitted, window)

                started = perf_counter()
                prediction = fitted.predict(futr_frame, quantiles=plan.quantiles)
                predict_seconds = perf_counter() - started

                prediction = _normalise_prediction(prediction, model.model_id, plan)
                _assert_prediction_after_cutoff(prediction, window, model.model_id)
                _validate_prediction(prediction, window, train.ids(), panel.spec, model.model_id)
            except LeakageError:
                # La fuga no es un fallo de modelo: es un run invalido.
                raise
            except Exception as exc:  # A6: los fallos ocupan una fila, no desaparecen
                records.append(
                    _record(
                        model,
                        window,
                        status=_FAILED,
                        error=f"{type(exc).__name__}: {exc}",
                        refit=refit,
                        refit_every=refit_every,
                        fit_seconds=fit_seconds,
                        predict_seconds=predict_seconds,
                        fitted=fitted,
                    )
                )
                windows_since_fit += 1
                continue

            prediction, crossings = _repair_quantile_crossing(prediction, plan.quantile_columns)
            forecast_parts.append(
                _assemble(prediction, panel, window, model.model_id, plan.quantile_columns)
            )
            records.append(
                _record(
                    model,
                    window,
                    status=_OK,
                    error=None,
                    refit=refit,
                    refit_every=refit_every,
                    fit_seconds=fit_seconds,
                    predict_seconds=predict_seconds,
                    fitted=fitted,
                    crossings=crossings,
                )
            )
            windows_since_fit += 1

    return BacktestResult(
        forecasts=_concat_forecasts(forecast_parts, plan),
        windows=_windows_frame(panel, windows),
        model_runs=_model_runs_frame(records),
        spec=panel.spec,
        plan=plan,
        futr_vintage=None if futr is None else futr.vintage,
    )


# --------------------------------------------------------------------------- #
# Barreras estructurales
# --------------------------------------------------------------------------- #


def _require_futr_provider(
    models: Sequence[Forecaster],
    spec: PanelSpec,
    futr: "FutrProvider | None",
) -> None:
    """Aborta el run si algun modelo necesita exogenas futuras que no existen.

    Falla antes de ajustar nada. La alternativa —evaluar el modelo sin ellas— lo
    dejaria compitiendo en el leaderboard bajo condiciones distintas de las
    declaradas, que es una comparacion mentirosa sin sintoma visible.

    Parameters
    ----------
    models
        Modelos del run.
    spec
        Especificacion del panel.
    futr
        Proveedor de exogenas futuras, si lo hay.

    Raises
    ------
    MissingFutrExog
        Si algun modelo declara `needs_futr_exog` y el run no tiene proveedor, o
        el panel no declara ninguna columna `futr_exog`.
    """
    needy = [model.model_id for model in models if model.requires.needs_futr_exog]
    if not needy:
        return
    if futr is None:
        raise MissingFutrExog(f"{needy} declaran needs_futr_exog y el run no tiene FutrProvider")
    if not spec.futr_exog:
        raise MissingFutrExog(
            f"{needy} declaran needs_futr_exog y el panel '{spec.dataset_id}' "
            "no declara ninguna columna futr_exog"
        )


def _assert_train_at_cutoff(train: Panel, window: Window) -> None:
    """Comprueba que el entrenamiento de una ventana no cruza su cutoff.

    Es la barrera que hace estructural la garantia de escalado: si lo que llega a
    `fit` no contiene ningun instante posterior al cutoff, ningun estadistico
    ajustado dentro de `fit` puede haberlo visto.

    Parameters
    ----------
    train
        Rebanada de entrenamiento producida por `Panel.train`.
    window
        Ventana en curso.

    Raises
    ------
    CutoffViolation
        Si el entrenamiento contiene instantes posteriores al cutoff.
    ChronolabError
        Si el entrenamiento esta vacio: predecir sin historia no es un caso
        degenerado admisible, es un plan mal construido.
    """
    if train.df.empty:
        raise ChronolabError(
            f"la ventana {window.window_id} no tiene datos de entrenamiento en "
            f"[{window.train_start}, {window.cutoff}]"
        )
    last = train.last_ds
    if last > window.cutoff:
        raise CutoffViolation(
            f"el entrenamiento de la ventana {window.window_id} llega a {last}, "
            f"posterior al cutoff {window.cutoff}"
        )


def _assert_fitted_at_or_before_cutoff(fitted: FittedForecaster, window: Window) -> None:
    """Comprueba que un ajuste reutilizado no es posterior a su ventana.

    Parameters
    ----------
    fitted
        Modelo ajustado, posiblemente en una ventana anterior.
    window
        Ventana en curso.

    Raises
    ------
    CutoffViolation
        Si el ajuste conoce informacion posterior al cutoff de la ventana.
    """
    if fitted.cutoff > window.cutoff:
        raise CutoffViolation(
            f"{fitted.model_id} esta ajustado hasta {fitted.cutoff}, posterior al "
            f"cutoff {window.cutoff} de la ventana {window.window_id}"
        )


def _assert_futr_frame(futr_frame: FutrFrame, window: Window, spec: PanelSpec) -> None:
    """Verifica la trama de exogenas futuras que entrega el proveedor.

    El `FutrFrame` ya impide leer una `hist_exog` por ausencia fisica, pero el
    motor no da por bueno lo que le llega de fuera: comprobar aqui que las
    columnas son exactamente las declaradas convierte un proveedor mal escrito en
    un run detenido en vez de en un leaderboard optimista.

    Parameters
    ----------
    futr_frame
        Trama emitida por el `FutrProvider`.
    window
        Ventana en curso.
    spec
        Especificacion del panel.

    Raises
    ------
    LeakageError
        Si aparece la columna objetivo o una `hist_exog`, o si algun instante no
        es posterior al cutoff.
    ChronolabError
        Si la trama no corresponde a la ventana o le faltan columnas declaradas.
    """
    if futr_frame.window != window:
        raise ChronolabError(
            f"el proveedor ha devuelto exogenas de la ventana {futr_frame.window.window_id} "
            f"para la ventana {window.window_id}"
        )

    columns = set(futr_frame.df.columns)
    forbidden = columns & ({spec.target} | set(spec.hist_exog))
    if forbidden:
        raise LeakageError(
            f"las exogenas futuras de la ventana {window.window_id} incluyen "
            f"{sorted(forbidden)}, que solo se conocen hasta el cutoff"
        )
    missing = set(spec.futr_exog) - columns
    if missing:
        raise ChronolabError(
            f"faltan exogenas futuras declaradas en la ventana {window.window_id}: "
            f"{sorted(missing)}"
        )

    if not futr_frame.df.empty and futr_frame.df["ds"].min() <= window.cutoff:
        raise CutoffViolation(
            f"las exogenas futuras de la ventana {window.window_id} llegan hasta "
            f"{futr_frame.df['ds'].min()}, anterior o igual al cutoff {window.cutoff}"
        )


def _assert_prediction_after_cutoff(
    prediction: pd.DataFrame, window: Window, model_id: ModelId
) -> None:
    """Comprueba que ninguna prediccion cae en el pasado ya conocido.

    Es la asercion central anti-fuga y esta en el unico camino que produce
    predicciones. Se ejecuta siempre: su coste es despreciable frente a su valor.

    Parameters
    ----------
    prediction
        Prediccion normalizada.
    window
        Ventana en curso.
    model_id
        Modelo que la produjo, para el mensaje.

    Raises
    ------
    CutoffViolation
        Si alguna ``ds`` es anterior o igual al cutoff.
    """
    if prediction.empty:
        return
    earliest = prediction["ds"].min()
    if earliest <= window.cutoff:
        raise CutoffViolation(
            f"{model_id} predice {earliest} en la ventana {window.window_id}, "
            f"anterior o igual a su cutoff {window.cutoff}"
        )


# --------------------------------------------------------------------------- #
# Contrato de la prediccion
# --------------------------------------------------------------------------- #


def _normalise_prediction(
    prediction: pd.DataFrame, model_id: ModelId, plan: BacktestPlan
) -> pd.DataFrame:
    """Lleva la salida de un modelo al esquema del artefacto.

    Los modelos con ``supports_quantiles=False`` no devuelven columnas de
    cuantil; se anaden como `NaN`. Nunca se inventa un intervalo: una banda
    fabricada por el arnes seria indistinguible de una estimada por el modelo.

    Parameters
    ----------
    prediction
        Salida cruda de `FittedForecaster.predict`.
    model_id
        Modelo que la produjo, para el mensaje.
    plan
        Plan del run, que fija la rejilla de cuantiles.

    Returns
    -------
    pandas.DataFrame
        Columnas ``unique_id``, ``ds``, ``y_hat`` y una por cuantil del plan.

    Raises
    ------
    PredictionContractError
        Si faltan columnas obligatorias o si ``ds`` no es UTC ingenuo.
    """
    missing = {"unique_id", "ds", "y_hat"} - set(prediction.columns)
    if missing:
        raise PredictionContractError(
            f"{model_id} no ha devuelto las columnas obligatorias {sorted(missing)}"
        )

    frame = prediction.copy()
    if not pd.api.types.is_datetime64_any_dtype(frame["ds"]):
        raise PredictionContractError(f"{model_id} ha devuelto 'ds' que no es datetime")
    if isinstance(frame["ds"].dtype, pd.DatetimeTZDtype):
        raise PredictionContractError(
            f"{model_id} ha devuelto 'ds' con huso horario; el panel es UTC ingenuo (I2)"
        )

    for column in plan.quantile_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["unique_id"] = frame["unique_id"].astype(str)
    return frame[["unique_id", "ds", "y_hat", *plan.quantile_columns]]


def _validate_prediction(
    prediction: pd.DataFrame,
    window: Window,
    ids: tuple[str, ...],
    spec: PanelSpec,
    model_id: ModelId,
) -> None:
    """Comprueba que la prediccion cubre exactamente el tramo evaluado.

    Un modelo que devuelve de mas, de menos o descolocado no puede compararse con
    los demas sobre los mismos instantes, y un `merge` posterior lo disimularia
    convirtiendo el desajuste en `NaN` silenciosos.

    Parameters
    ----------
    prediction
        Prediccion normalizada.
    window
        Ventana en curso.
    ids
        Series del entrenamiento de la ventana.
    spec
        Especificacion del panel, que fija la frecuencia de la rejilla.
    model_id
        Modelo que la produjo, para el mensaje.

    Raises
    ------
    PredictionContractError
        Si sobran o faltan series, si hay instantes fuera de la rejilla evaluada,
        si hay duplicados o si el numero de filas no es ``len(ids) * h``.
    """
    expected_grid = pd.date_range(window.first_pred, window.last_pred, freq=spec.freq)

    predicted_ids = set(prediction["unique_id"])
    expected_ids = {str(uid) for uid in ids}
    if predicted_ids != expected_ids:
        raise PredictionContractError(
            f"{model_id} ha predicho las series {sorted(predicted_ids)} y la ventana "
            f"{window.window_id} evalua {sorted(expected_ids)}"
        )

    outside = set(prediction["ds"]) - set(expected_grid)
    if outside:
        raise PredictionContractError(
            f"{model_id} ha predicho {len(outside)} instantes fuera de "
            f"[{window.first_pred}, {window.last_pred}] en la ventana {window.window_id}"
        )
    if prediction.duplicated(subset=["unique_id", "ds"]).any():
        raise PredictionContractError(
            f"{model_id} ha devuelto filas duplicadas por (unique_id, ds) en la "
            f"ventana {window.window_id}"
        )

    expected_rows = len(expected_ids) * window.h
    if len(prediction) != expected_rows:
        raise PredictionContractError(
            f"{model_id} ha devuelto {len(prediction)} filas y la ventana "
            f"{window.window_id} exige {expected_rows} ({len(expected_ids)} series x "
            f"h={window.h})"
        )


def _repair_quantile_crossing(
    prediction: pd.DataFrame, quantile_columns: tuple[str, ...]
) -> tuple[pd.DataFrame, int]:
    """Ordena los cuantiles cruzados y cuenta cuantas filas hubo que reparar.

    Ordenar es correcto —es una proyeccion isotonica trivial— pero la frecuencia
    del cruce es un diagnostico del modelo, asi que se cuenta en lugar de
    arreglarse en silencio.

    Un modelo que solo calibra algunos niveles —dos bandas conformales en vez
    de los siete cuantiles canonicos, por ejemplo— deja `NaN` en las columnas
    que no calculo (docs/ARCHITECTURE.md §7.3: nunca se inventa un intervalo).
    Esas `NaN` no participan en la comprobacion de cruce ni en el reordenado:
    ``numpy.sort`` trata `NaN` como el valor mas grande y las desplazaria al
    final de la fila, pisando un cuantil real que si se calculo. Aqui se
    ordenan solo los valores finitos de cada fila, cada uno de vuelta a una
    posicion que ya era finita; las columnas sin calcular siguen siendo `NaN`.

    Parameters
    ----------
    prediction
        Prediccion normalizada.
    quantile_columns
        Columnas de cuantil, en orden creciente de cuantil.

    Returns
    -------
    tuple
        Prediccion con los cuantiles ordenados y numero de filas reparadas.
    """
    if len(quantile_columns) < 2:
        return prediction, 0

    values = prediction[list(quantile_columns)].to_numpy(dtype=float)
    ordered = values.copy()
    crossed = np.zeros(len(values), dtype=bool)

    for row in range(values.shape[0]):
        finite = np.isfinite(values[row])
        if finite.sum() < 2:
            continue
        current = values[row, finite]
        sorted_values = np.sort(current)
        if not np.array_equal(current, sorted_values):
            ordered[row, finite] = sorted_values
            crossed[row] = True

    n_crossed = int(crossed.sum())
    if n_crossed:
        repaired = prediction.copy()
        repaired[list(quantile_columns)] = ordered
        return repaired, n_crossed
    return prediction, 0


# --------------------------------------------------------------------------- #
# Ensamblado de artefactos
# --------------------------------------------------------------------------- #


def _assemble(
    prediction: pd.DataFrame,
    panel: Panel,
    window: Window,
    model_id: ModelId,
    quantile_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Une prediccion y observacion en filas de la tabla `forecasts`.

    Parameters
    ----------
    prediction
        Prediccion validada de una ventana.
    panel
        Panel completo, del que se toman los valores observados del tramo.
    window
        Ventana en curso.
    model_id
        Modelo que produjo la prediccion.
    quantile_columns
        Columnas de cuantil del plan.

    Returns
    -------
    pandas.DataFrame
        Con las columnas de `FORECAST_KEY_COLUMNS` mas las de cuantil.
    """
    actuals = panel.actuals(window).rename(columns={panel.spec.target: "y"})
    actuals["unique_id"] = actuals["unique_id"].astype(str)
    frame = prediction.merge(actuals, on=["unique_id", "ds"], how="left")

    grid = pd.date_range(window.first_pred, window.last_pred, freq=panel.spec.freq)
    frame["h_step"] = grid.get_indexer(pd.DatetimeIndex(frame["ds"])) + 1
    frame["model_id"] = str(model_id)
    frame["window_id"] = window.window_id
    frame["cutoff"] = window.cutoff

    return _cast_forecasts(frame[[*FORECAST_KEY_COLUMNS, *quantile_columns]])


def _cast_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Fija los tipos de la tabla `forecasts` (docs/ARCHITECTURE.md §7.4).

    Parameters
    ----------
    frame
        Trama con las columnas correctas y tipos sin normalizar.

    Returns
    -------
    pandas.DataFrame
        Con `window_id` y `h_step` en `int16`, valores en `float32` y marcas de
        tiempo en `datetime64[ns]`.
    """
    typed = frame.copy()
    typed["window_id"] = typed["window_id"].astype("int16")
    typed["h_step"] = typed["h_step"].astype("int16")
    for column in typed.columns:
        if column in ("y", "y_hat") or column.startswith("q_"):
            typed[column] = typed[column].astype("float32")
    typed["ds"] = typed["ds"].astype("datetime64[ns]")
    typed["cutoff"] = typed["cutoff"].astype("datetime64[ns]")
    return typed


def _concat_forecasts(parts: list[pd.DataFrame], plan: BacktestPlan) -> pd.DataFrame:
    """Une los tramos de `forecasts` y fija el orden fisico de las filas.

    Parameters
    ----------
    parts
        Tramos por ``(modelo, ventana)``.
    plan
        Plan del run, que fija las columnas de cuantil.

    Returns
    -------
    pandas.DataFrame
        Ordenada por ``(model_id, unique_id, window_id, ds)``, que es el orden en
        que se escribe en parquet para que el filtrado por serie lea solo los
        grupos de fila necesarios.
    """
    columns = [*FORECAST_KEY_COLUMNS, *plan.quantile_columns]
    if not parts:
        empty = pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
        empty["ds"] = pd.Series(dtype="datetime64[ns]")
        empty["cutoff"] = pd.Series(dtype="datetime64[ns]")
        empty["window_id"] = pd.Series(dtype="int16")
        empty["h_step"] = pd.Series(dtype="int16")
        for column in columns:
            if column in ("y", "y_hat") or column.startswith("q_"):
                empty[column] = pd.Series(dtype="float32")
        return empty[columns]

    frame = pd.concat(parts, ignore_index=True)
    return frame.sort_values(["model_id", "unique_id", "window_id", "ds"]).reset_index(drop=True)


def _windows_frame(panel: Panel, windows: tuple[Window, ...]) -> pd.DataFrame:
    """Construye la tabla `windows` del run.

    Parameters
    ----------
    panel
        Panel evaluado.
    windows
        Ventanas efectivas del run.

    Returns
    -------
    pandas.DataFrame
        Una fila por ventana, con el tamano real de su entrenamiento. `n_obs` es
        lo que delata a una ventana que parecia larga y estaba llena de huecos.
    """
    rows: list[dict[str, object]] = []
    for window in windows:
        train = panel.train(window)
        observed = train.df[panel.spec.target].notna()
        rows.append(
            {
                "window_id": window.window_id,
                "stage": window.stage,
                "train_start": window.train_start,
                "cutoff": window.cutoff,
                "first_pred": window.first_pred,
                "last_pred": window.last_pred,
                "n_train_obs": int(observed.sum()),
                "n_series": int(train.df.loc[observed, "unique_id"].nunique()),
            }
        )

    frame = pd.DataFrame(rows)
    frame["window_id"] = frame["window_id"].astype("int16")
    frame["n_train_obs"] = frame["n_train_obs"].astype("int64")
    frame["n_series"] = frame["n_series"].astype("int32")
    return frame


def _record(
    model: Forecaster,
    window: Window,
    *,
    status: str,
    error: str | None,
    refit: bool,
    refit_every: int,
    fit_seconds: float = 0.0,
    predict_seconds: float = 0.0,
    fitted: FittedForecaster | None = None,
    crossings: int = 0,
) -> dict[str, object]:
    """Fila de `model_runs` para un par ``(modelo, ventana)``.

    Parameters
    ----------
    model
        Modelo evaluado.
    window
        Ventana en curso.
    status
        ``"ok"``, ``"failed"`` o ``"skipped"``.
    error
        Tipo y mensaje del fallo, o ``None``.
    refit
        Si se reajusto en esta ventana.
    refit_every
        Politica aplicada.
    fit_seconds, predict_seconds
        Coste medido. Cero cuando se reutiliza un ajuste.
    fitted
        Ajuste vigente, del que se toma `n_params`.
    crossings
        Filas con cuantiles cruzados reparadas.

    Returns
    -------
    dict
        Fila lista para `_model_runs_frame`.
    """
    return {
        "model_id": str(model.model_id),
        "window_id": window.window_id,
        "status": status,
        "error": error,
        "refit": refit,
        "refit_every": refit_every,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_params": None if fitted is None else fitted.n_params,
        "is_zero_shot": model.requires.is_zero_shot,
        "quantile_crossings": crossings,
    }


def _model_runs_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Construye la tabla `model_runs` del run.

    Parameters
    ----------
    records
        Filas producidas por `_record`.

    Returns
    -------
    pandas.DataFrame
        Una fila por ``(modelo, ventana)``, incluidas las fallidas y las
        saltadas: sin ellas, las metricas de un modelo que solo sobrevivio a la
        mitad de las ventanas pareceria comparable con las del resto.
    """
    columns = [
        "model_id",
        "window_id",
        "status",
        "error",
        "refit",
        "refit_every",
        "fit_seconds",
        "predict_seconds",
        "n_params",
        "is_zero_shot",
        "quantile_crossings",
    ]
    if not records:  # pragma: no cover  el motor siempre registra algo
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})

    frame = pd.DataFrame(records)[columns]
    frame["window_id"] = frame["window_id"].astype("int16")
    frame["refit_every"] = frame["refit_every"].astype("int16")
    frame["fit_seconds"] = frame["fit_seconds"].astype("float32")
    frame["predict_seconds"] = frame["predict_seconds"].astype("float32")
    frame["n_params"] = frame["n_params"].astype("Int64")
    frame["refit"] = frame["refit"].astype("bool")
    frame["is_zero_shot"] = frame["is_zero_shot"].astype("bool")
    frame["quantile_crossings"] = frame["quantile_crossings"].astype("int32")
    return frame
