"""Unica ruta de lectura de artefactos. Es la API que consume la app.

Tambien es el unico constructor de `ScoringFrame`, lo que garantiza que un
detector jamas reciba predicciones dentro de muestra (fuga L9).

Alcance actual: la variante **en memoria**, que construye el `ScoringFrame` a
partir de un `BacktestResult` sin pasar por parquet. La lectura desde disco
comparte esta funcion en cuanto exista el escritor: lo que cambia es de donde
salen las tablas, no como se recorta el tramo.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from chronolab.anomaly.protocols import ScoringFrame
from chronolab.errors import ArtifactNotFound, PredictionContractError, WindowValidationError
from chronolab.types import ModelId, Stage

if TYPE_CHECKING:  # pragma: no cover
    # Import diferido: en tiempo de ejecucion aqui solo se leen atributos del
    # resultado, asi que `artifacts` no depende de `evaluation`. Es el mismo
    # patron que usa el motor con `FutrProvider`.
    from chronolab.evaluation.backtest import BacktestResult

__all__ = ["SCORING_FRAME_COLUMNS", "scoring_frame"]

SCORING_FRAME_COLUMNS: tuple[str, ...] = (
    "unique_id",
    "ds",
    "y",
    "y_hat",
    "cutoff",
    "h_step",
)
"""Columnas de `ScoringFrame.df` anteriores a las de cuantil."""


def scoring_frame(
    result: "BacktestResult",
    *,
    model_id: ModelId,
    stage: Stage,
) -> ScoringFrame:
    """Tramo puntuable de un run, para un modelo y una etapa.

    Es el unico camino por el que un detector recibe datos. Como las filas salen
    de la tabla `forecasts`, que el motor solo escribe con predicciones
    posteriores al cutoff de su ventana, ningun detector puede ver un residuo
    dentro de muestra: no existe codigo que se lo entregue.

    El corte entre calibracion y puntuacion es la frontera ``dev`` / ``holdout``
    que el splitter ya emite. No hay parametro nuevo, y de ahi salen tres
    propiedades:

    1. El corte cae **entre** ventanas, nunca dentro de una. Partir dentro
       dejaria residuos del mismo origen de prediccion a ambos lados; comparten
       ajuste y estan fuertemente correlados, de modo que el cuantil de
       calibracion se estimaria en parte con la misma prediccion que despues se
       puntua.
    2. Las ventanas ``holdout`` son siempre las ultimas del plan, asi que
       ``calib.end < frame.start`` sin comprobar nada.
    3. La disciplina de tuning se hereda: lo que se ajuste sobre ``dev`` no ha
       visto ``holdout``.

    Parameters
    ----------
    result
        Artefactos en memoria de un run de backtesting.
    model_id
        Modelo del que se toman ``y_hat`` y los cuantiles.
    stage
        ``"dev"`` para el tramo de calibracion, ``"holdout"`` para el que se
        puntua.

    Returns
    -------
    ScoringFrame
        Con rejilla completa a ``spec.freq`` sobre el tramo evaluado de las
        ventanas de esa etapa, una fila por ``(unique_id, ds)`` y ordenada por
        esa misma clave. Las ventanas en las que el modelo fallo o se salto
        dejan `NaN` explicito y `cutoff` a `NaT`, nunca filas ausentes.

    Raises
    ------
    WindowValidationError
        Si el plan del run no admite deteccion: ``step_size != h`` haria que los
        tramos evaluados se solapasen —y entonces un instante tendria varias
        predicciones y varias etapas— y ``holdout_windows == 0`` dejaria el
        tramo de puntuacion vacio.
    ArtifactNotFound
        Si el run no tiene ese modelo, o no tiene ninguna ventana de esa etapa.
    PredictionContractError
        Si el tramo trae dos filas para el mismo ``(unique_id, ds)``, que es lo
        que el teselado del plan tiene que impedir.
    """
    plan = result.plan
    if plan.step_size != plan.h:
        raise WindowValidationError(
            f"la deteccion de anomalias exige un plan teselado (step_size == h) y este run "
            f"tiene step_size={plan.step_size} con h={plan.h}: con solape, un instante "
            f"tendria varias predicciones y podria caer en dev y en holdout a la vez"
        )
    if plan.holdout_windows < 1:
        raise WindowValidationError(
            "la deteccion de anomalias exige al menos una ventana de holdout: sin ella no "
            "hay tramo que puntuar despues de calibrar"
        )

    windows = result.windows
    stage_windows = windows.loc[windows["stage"] == stage]
    if stage_windows.empty:
        raise ArtifactNotFound(f"el run no tiene ninguna ventana con stage='{stage}'")

    forecasts = result.forecasts
    of_model = forecasts.loc[forecasts["model_id"] == str(model_id)]
    if of_model.empty:
        raise ArtifactNotFound(f"el run no tiene predicciones del modelo '{model_id}'")

    rows = of_model.loc[of_model["window_id"].isin(stage_windows["window_id"])]
    if rows.duplicated(subset=["unique_id", "ds"]).any():
        raise PredictionContractError(
            f"el tramo '{stage}' del modelo '{model_id}' trae varias filas para el mismo "
            "(unique_id, ds); el plan teselado deberia haberlo impedido"
        )

    start = pd.Timestamp(stage_windows["first_pred"].min())
    end = pd.Timestamp(stage_windows["last_pred"].max())
    grid = pd.date_range(start, end, freq=result.spec.freq)
    # Las series salen de todas las ventanas del modelo, no solo de las de esta
    # etapa: una serie que el modelo no pudo predecir en holdout tiene que
    # aparecer con `NaN`, no desaparecer del tramo.
    ids = sorted({str(uid) for uid in of_model["unique_id"].unique()})

    quantile_columns = [name for name in forecasts.columns if name.startswith("q_")]
    columns = [*SCORING_FRAME_COLUMNS, *quantile_columns]
    frame = _on_complete_grid(rows, ids=ids, grid=grid, columns=columns)

    return ScoringFrame(
        df=frame,
        spec=result.spec,
        model_id=model_id,
        start=start,
        end=end,
    )


def _on_complete_grid(
    rows: pd.DataFrame,
    *,
    ids: list[str],
    grid: pd.DatetimeIndex,
    columns: list[str],
) -> pd.DataFrame:
    """Reindexa un tramo sobre el producto ``series x rejilla``.

    Una ventana fallida o saltada no deja filas en `forecasts`. Devolverlas
    ausentes convertiria el hueco en algo invisible y desalinearia la
    comparacion entre detectores; devolverlas como `NaN` explicito lo deja
    auditable y hace que el detector las marque no puntuables.

    Parameters
    ----------
    rows
        Filas de `forecasts` del modelo y la etapa.
    ids
        Series del run, ordenadas.
    grid
        Rejilla temporal completa del tramo.
    columns
        Columnas de salida, en orden.

    Returns
    -------
    pandas.DataFrame
        Con ``len(ids) * len(grid)`` filas, ordenada por ``(unique_id, ds)``.
    """
    index = pd.MultiIndex.from_product([ids, grid], names=["unique_id", "ds"])
    present = rows.copy()
    present["unique_id"] = present["unique_id"].astype(str)
    present = present.set_index(["unique_id", "ds"])

    available = [name for name in columns if name in present.columns]
    frame = present[available].reindex(index)
    for name in columns:
        if name not in frame.columns and name not in ("unique_id", "ds"):
            frame[name] = np.nan

    frame = frame.reset_index()
    # `Int16` y no `int16`: una ventana fallida deja el hueco, y un entero sin
    # nulo obligaria a inventarse un `h_step` que no existe.
    frame["h_step"] = frame["h_step"].astype("Float64").astype("Int16")
    frame["cutoff"] = frame["cutoff"].astype("datetime64[ns]")
    for name in frame.columns:
        if name in ("y", "y_hat") or name.startswith("q_"):
            frame[name] = frame[name].astype("float32")
    return frame[columns].sort_values(["unique_id", "ds"]).reset_index(drop=True)
