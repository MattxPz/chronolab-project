"""Tuning con Optuna, limitado por construccion a las ventanas de desarrollo del backtest.

La barrera contra la fuga L5 de docs/ARCHITECTURE.md ("tuning sobre las
ventanas de reporte") no es una convencion aqui: es **ausencia fisica**, la
mas fuerte de las tres formas admisibles (docs/ARCHITECTURE.md A3). `tune()`
nunca recibe el panel completo del run: `dev_only_panel()` lo recorta al
tramo que cubren las ventanas ``stage="dev"`` del plan *antes* de que exista
ningun trial, de modo que el `Forecaster` que construye cada trial —y por
tanto Optuna, que solo ve el numero que ese `Forecaster` produce— no tiene
forma de leer nada de las ventanas de holdout. No hace falta un tipo
`DevWindow` distinto de `Window`, ni una comprobacion en tiempo de ejecucion
que alguien podria olvidar: los datos de holdout, simplemente, no estan en la
estructura que recibe `backtest()` dentro del bucle de Optuna.

Presupuesto de trials configurable
------------------------------------
`n_trials` es un parametro de `tune()`, nunca una constante: un tuning con
LightGBM/XGBoost sobre varias ventanas de desarrollo cuesta ``n_trials *
n_dev_windows`` ajustes, y ese coste hay que poder acotarlo desde quien llama
—un `scripts/` de CI acotado a unos pocos trials, un analisis exploratorio
con muchos mas— sin tocar este modulo.

Import perezoso
----------------
`optuna` vive en el extra `ml` (D20), no en el nucleo. El modulo tiene que
poder **importarse** sin ese extra —lo exige `tests/unit/test_module_tree.py`—
asi que `import optuna` no aparece a nivel de modulo: vive dentro de
`_require_optuna`, llamada solo al invocar `tune()`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from chronolab.evaluation.aggregate import score_forecasts
from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.evaluation.metrics import point_metrics, probabilistic_metrics
from chronolab.models.protocols import Forecaster
from chronolab.panel import Panel

if TYPE_CHECKING:  # pragma: no cover
    # Import diferido: en tiempo de ejecucion `tune()` solo reenvia el
    # proveedor a `backtest()`, que ya lo trata como estructural
    # (`evaluation.backtest` no depende de `data` en tiempo de ejecucion, ver
    # su propio docstring). El mismo patron, aplicado aqui. `optuna.Trial` y
    # `optuna.Study` se tipan como `Any` (D16): son objetos de una libreria en
    # la cuarentena de tipos del proyecto.
    from chronolab.data.futr import FutrProvider

__all__ = ["Direction", "TuningResult", "dev_only_panel", "tune"]

Direction = Literal["minimize", "maximize"]
"""Sentido de la optimizacion. La mayoria de las metricas del proyecto
(``mase``, ``rmse``, ``pinball_mean``...) se minimizan; ninguna de las que
persiste `chronolab.evaluation.metrics` se maximiza hoy, pero el parametro
queda abierto para una metrica de cobertura o de ganancia que se anada despues.
"""


def _require_optuna() -> Any:
    """Importa `optuna` bajo demanda, con un mensaje util si falta el extra.

    Returns
    -------
    Any
        El modulo `optuna`. Tipado como `Any` a proposito (D16): `optuna` esta
        en la cuarentena de tipos del proyecto.

    Raises
    ------
    ImportError
        Si `optuna` no esta instalado.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra ml
        raise ImportError(
            "chronolab.evaluation.tuning necesita el extra 'ml' con optuna instalado: "
            "`uv sync --extra ml`."
        ) from exc
    return optuna


def dev_only_panel(panel: Panel, plan: BacktestPlan) -> tuple[Panel, BacktestPlan]:
    """Recorta un panel y reescala un plan a exactamente las ventanas ``stage="dev"``.

    Parameters
    ----------
    panel
        Panel canonico completo del run que se quiere tunear.
    plan
        Plan de backtesting completo, con `holdout_windows` ya fijado a las
        ventanas que se reservan para el reporte final.

    Returns
    -------
    tuple
        ``(panel_recortado, plan_dev)``. `panel_recortado` termina exactamente
        en el ultimo instante que evalua la ultima ventana `dev`: nada
        posterior a ese punto viaja en la estructura. `plan_dev` es una copia
        de `plan` con ``n_windows`` igual al numero de ventanas `dev` y
        ``holdout_windows=0``, de modo que volver a particionar
        `panel_recortado` con el reproduce exactamente esas mismas ventanas
        —misma aritmetica de `RollingOriginSplitter`, ancla en el nuevo final
        del panel— pero todas etiquetadas ``"dev"``.

    Raises
    ------
    WindowValidationError
        Si `panel` no da para el plan completo (delegado en el splitter).
    ValueError
        Si `plan` no reserva ninguna ventana `dev` —nada sobre lo que tunear.
    """
    windows = plan.splitter().split(panel)
    dev_windows = tuple(window for window in windows if window.stage == "dev")
    if not dev_windows:
        raise ValueError(
            f"el plan no reserva ninguna ventana 'dev' (holdout_windows={plan.holdout_windows} "
            f"de n_windows={plan.n_windows}): no hay nada sobre lo que tunear"
        )

    dev_end = max(window.last_pred for window in dev_windows)
    trimmed = panel.slice(panel.first_ds, dev_end)
    dev_plan = replace(plan, n_windows=len(dev_windows), holdout_windows=0)
    return trimmed, dev_plan


def _objective_value(
    scored: pd.DataFrame, plan: BacktestPlan, *, metric: str, direction: Direction
) -> float:
    """Extrae el valor de `metric` de un tramo ya puntuado de `forecasts`.

    Parameters
    ----------
    scored
        Salida de `chronolab.evaluation.aggregate.score_forecasts`: `forecasts`
        con el denominador de MASE ya unido.
    plan
        Plan `dev` del tuning, que fija la rejilla de cuantiles.
    metric
        Nombre de la metrica, cualquiera de las que devuelven
        `chronolab.evaluation.metrics.point_metrics` o `.probabilistic_metrics`
        (``"mase"``, ``"mae"``, ``"pinball_mean"``, ``"crps_discrete"``...).
    direction
        Sentido de la optimizacion, para saber que valor "centinela" devolver
        cuando el trial no produjo ninguna observacion valida.

    Returns
    -------
    float
        El valor de la metrica sobre las observaciones crudas del tramo `dev`
        —nunca el promedio de metricas ya agregadas por ventana, la misma
        regla que impone `chronolab.evaluation.aggregate`—, o el peor valor
        posible (``inf`` / ``-inf`` segun `direction`) si el tramo esta vacio
        o la metrica no esta definida. Un trial con hiperparametros que
        revientan el modelo en todas las ventanas dev no debe tirar el
        estudio entero: Optuna lo descarta como el peor trial posible, en vez
        de propagar la excepcion.
    """
    worst = math.inf if direction == "minimize" else -math.inf
    if scored.empty:
        return worst

    row: dict[str, float] = {
        **point_metrics(scored),
        **probabilistic_metrics(scored, quantiles=plan.quantiles),
    }
    value = row.get(metric)
    if value is None or not math.isfinite(value):
        return worst
    return float(value)


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Resultado de un tuning: el estudio de Optuna y el plan efectivo usado.

    Attributes
    ----------
    study
        Estudio de Optuna completo. `study.best_params` y `study.best_value`
        son la superficie que consulta quien llama; `study.trials` da el
        historial completo para un grafico de convergencia.
    dev_plan
        Plan de backtesting reescalado a las ventanas `dev`
        (`dev_only_panel`), el que de verdad se uso en cada trial.
    n_dev_windows
        Ventanas de desarrollo evaluadas por trial. Copiado de `dev_plan` por
        comodidad, para no tener que desempaquetarlo.
    """

    study: Any
    dev_plan: BacktestPlan
    n_dev_windows: int


def tune(
    panel: Panel,
    build_model: Callable[[Any], Forecaster],
    plan: BacktestPlan,
    *,
    n_trials: int = 20,
    metric: str = "mase",
    direction: Direction = "minimize",
    seed: int = 0,
    futr: FutrProvider | None = None,
    show_progress_bar: bool = False,
) -> TuningResult:
    """Optimiza un `Forecaster` con Optuna, viendo unicamente las ventanas `dev` del plan.

    Cada trial construye un modelo con `build_model(trial)`, lo evalua con
    `chronolab.evaluation.backtest.backtest` sobre el panel recortado que
    devuelve `dev_only_panel`, y puntua el resultado con `metric`. El plan que
    de verdad ve cada trial no es `plan`: es su version `dev_only_panel`, con
    ``n_windows`` igual al numero de ventanas `dev` de `plan` y
    ``holdout_windows=0``.

    Parameters
    ----------
    panel
        Panel canonico completo del run. Solo se usa para calcular el tramo
        `dev`; ningun trial recibe este objeto.
    build_model
        Construye un `Forecaster` a partir de un `optuna.Trial`, tipicamente
        sugiriendo hiperparametros con ``trial.suggest_*`` y pasandolos al
        constructor del modelo (por ejemplo, ``params=`` de
        `chronolab.models.adapters.mlforecast.LightGBMForecaster`).
    plan
        Plan de backtesting completo del run, con `holdout_windows` fijado a
        las ventanas que se reservan para el reporte final.
    n_trials
        Presupuesto de trials. Configurable a proposito (ver el docstring del
        modulo): el coste es ``n_trials`` ajustes por ventana `dev`.
    metric
        Metrica a optimizar. Cualquiera de las que devuelven
        `chronolab.evaluation.metrics.point_metrics` o `.probabilistic_metrics`.
        ``"mase"`` por defecto: la metrica principal del proyecto.
    direction
        ``"minimize"`` (por defecto) o ``"maximize"``.
    seed
        Semilla del muestreador de Optuna (`TPESampler`), para que el orden de
        exploracion de los trials sea reproducible.
    futr
        Proveedor de exogenas futuras, si algun modelo del espacio de busqueda
        lo necesita. Si esta construido sobre un `Panel`, debe construirse
        sobre el panel recortado que devuelve `dev_only_panel(panel, plan)`,
        no sobre `panel`: pasar aqui un proveedor atado al panel completo
        reintroduce por otra puerta la fuga que este modulo existe para
        impedir.
    show_progress_bar
        Barra de progreso de Optuna en la terminal.

    Returns
    -------
    TuningResult

    Raises
    ------
    ImportError
        Si `optuna` no esta instalado.
    ValueError
        Si `plan` no reserva ninguna ventana `dev`.
    WindowValidationError
        Si `panel` no da para `plan`.
    """
    optuna = _require_optuna()
    trimmed_panel, dev_plan = dev_only_panel(panel, plan)

    def objective(trial: Any) -> float:
        model = build_model(trial)
        result = backtest(trimmed_panel, [model], dev_plan, futr=futr)
        scored = score_forecasts(result, trimmed_panel, stage=None)
        return _objective_value(scored, dev_plan, metric=metric, direction=direction)

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return TuningResult(study=study, dev_plan=dev_plan, n_dev_windows=dev_plan.n_windows)
