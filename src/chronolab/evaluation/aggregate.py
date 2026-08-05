"""Reglas de rollup de `forecasts` a `metrics`, con marginalizacion explicita.

Dos reglas innegociables: toda fila se calcula desde `forecasts` (nunca se
agrega un agregado, porque MASE y sMAPE son cocientes) y no se comparan filas
con distinto `futr_vintage`.

La primera regla es la que gobierna este modulo entero. La fila "modelo X sobre
todas las series" **no** es el promedio de las filas "modelo X sobre la serie
s00", "…s01", …: es el mismo calculo repetido sobre el conjunto crudo de
observaciones. Con MAE dan lo mismo si todas las series tienen el mismo numero de
puntos; con MASE, sMAPE, MAPE y la cobertura, no. Encadenar agregaciones produce
numeros que parecen razonables y no significan nada, y es la forma mas comun de
publicar un leaderboard equivocado.

Nota de alcance: `build_leaderboard` escribe una tabla **ancha**, comoda de leer
y de ordenar, en `reports/results/`. La fuente de verdad del proyecto sigue
siendo el esquema en estrella de docs/ARCHITECTURE.md §7 —en particular la tabla
larga `metrics.parquet`, que admite metricas nuevas sin migrar el esquema y
marginaliza por serie, ventana y paso—; el leaderboard es una vista derivada y
comoda, no su sustituto. Mientras `artifacts/` no exista, este fichero es el
resultado publicable de un run, y por eso lleva `stage` y `n_obs` en cada fila:
sin ellos no se puede saber sobre que se calculo.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from chronolab.errors import UnstableMetricWarning
from chronolab.evaluation.backtest import BacktestResult
from chronolab.evaluation.metrics import mase_denominators, point_metrics, probabilistic_metrics
from chronolab.panel import Panel
from chronolab.types import Stage

__all__ = [
    "DEFAULT_LEADERBOARD_PATH",
    "build_leaderboard",
    "score_forecasts",
    "select_stage",
]

DEFAULT_LEADERBOARD_PATH = Path("reports/results/leaderboard.parquet")
"""Ruta por defecto del leaderboard, relativa al directorio de trabajo.

El proyecto se ejecuta desde la raiz del repositorio (`make test`, `make app`),
y `reports/` esta versionado con un tope de tamano en CI. La resolucion de rutas
con configuracion propia vive en `chronolab.config`, que todavia no la ofrece.
"""

_TIMING_COLUMNS: tuple[str, ...] = (
    "fit_seconds_total",
    "fit_seconds_mean",
    "predict_seconds_total",
    "predict_seconds_mean",
    "n_refits",
    "n_windows_ok",
    "n_windows_failed",
    "n_windows_skipped",
    "is_zero_shot",
)


def select_stage(
    forecasts: pd.DataFrame,
    windows: pd.DataFrame,
    stage: Stage | None,
) -> pd.DataFrame:
    """Recorta `forecasts` a las ventanas de una etapa.

    La tabla de predicciones no lleva la etapa: la lleva `windows`, y la union se
    hace por `window_id`. Es deliberado —desnormalizar `stage` en la tabla grande
    no aporta nada que la union no de— pero obliga a que el corte pase siempre
    por aqui.

    Parameters
    ----------
    forecasts
        Tabla `forecasts` de un run.
    windows
        Tabla `windows` del mismo run.
    stage
        ``"holdout"``, ``"dev"``, o ``None`` para no filtrar.

    Returns
    -------
    pandas.DataFrame
        Las filas de las ventanas de esa etapa. Puede quedar vacia si el plan no
        reservo ninguna: un run sin holdout no tiene nada que publicar, y eso
        debe verse como una tabla vacia y no como un leaderboard silencioso sobre
        ventanas de desarrollo.
    """
    if stage is None:
        return forecasts
    selected = windows.loc[windows["stage"] == stage, "window_id"]
    return forecasts[forecasts["window_id"].isin(selected)]


def score_forecasts(
    result: BacktestResult,
    panel: Panel,
    *,
    stage: Stage | None = None,
    season: int | None = None,
) -> pd.DataFrame:
    """Une a `forecasts` el denominador de MASE de cada serie y ventana.

    Parameters
    ----------
    result
        Resultado del backtest.
    panel
        Panel canonico del run, del que sale el entrenamiento de cada ventana.
    stage
        Etapa a conservar. Ver `select_stage`.
    season
        Longitud estacional del denominador. Por defecto, la del panel.

    Returns
    -------
    pandas.DataFrame
        `forecasts` con una columna ``mase_denominator``, calculada **con el
        entrenamiento de cada ventana** y unida por ``(unique_id, window_id)``.
        Nunca hay un denominador global: cada fila lleva el de su ventana.
    """
    forecasts = select_stage(result.forecasts, result.windows, stage)
    if forecasts.empty:
        return forecasts.assign(mase_denominator=pd.Series(dtype="float64"))

    windows = result.windows[result.windows["window_id"].isin(forecasts["window_id"].unique())]
    denominators = mase_denominators(panel, windows, season=season)
    return forecasts.merge(denominators, on=["unique_id", "window_id"], how="left")


def build_leaderboard(
    result: BacktestResult,
    panel: Panel,
    *,
    stage: Stage | None = None,
    season: int | None = None,
    quantiles: Sequence[float] | None = None,
    include_overall: bool = True,
    path: Path | None = DEFAULT_LEADERBOARD_PATH,
) -> pd.DataFrame:
    """Agrega un run a una tabla por modelo y serie, y la persiste.

    Cada fila reune la calidad de las predicciones y su coste: sin las columnas
    de tiempo, un modelo que tarda cuatro horas y otro que tarda cuatro segundos
    se leen igual, y el eje precision-coste es la mitad del interes del proyecto.

    Las filas con ``unique_id`` nulo son el agregado del modelo sobre todas sus
    series. **No** son el promedio de las filas por serie: se recalculan desde
    las observaciones crudas, porque promediar MASE, sMAPE o cobertura por series
    produce un numero que no es ninguna de las dos cosas.

    Los tiempos son una propiedad del par ``(modelo, ventana)``, no de la serie:
    un ajuste cubre todas las series a la vez. Por eso las columnas de coste
    valen lo mismo en todas las filas de un modelo, y estan repetidas para que la
    tabla se pueda ordenar por coste sin unir con `model_runs`.

    Parameters
    ----------
    result
        Resultado del backtest.
    panel
        Panel canonico del run.
    stage
        Etapa a evaluar. ``None`` usa todas las ventanas y escribe ``"all"`` en
        la columna `stage`. Para publicar, ``"holdout"``: el leaderboard que se
        ensena no debe contener las ventanas sobre las que se ajustaron
        decisiones.
    season
        Longitud estacional del denominador de MASE. Por defecto, la del panel.
    quantiles
        Rejilla de cuantiles a evaluar. Por defecto, la del plan del run.
    include_overall
        Anadir la fila agregada por modelo, con ``unique_id`` nulo.
    path
        Destino del parquet. ``None`` no escribe nada. La escritura es atomica
        —fichero temporal y renombrado— para que nadie lea media tabla.

    Returns
    -------
    pandas.DataFrame
        Ordenada con los agregados por modelo primero y, dentro de cada bloque,
        por MASE creciente: el orden en que se mira un leaderboard.

    Warns
    -----
    UnstableMetricWarning
        Una sola vez, resumido, si MAPE resulto inestable en alguna combinacion
        de modelo y serie. El detalle por serie lo da `metrics.mape` cuando se
        llama directamente.
    """
    grid = tuple(result.plan.quantiles) if quantiles is None else tuple(quantiles)
    scored = score_forecasts(result, panel, stage=stage, season=season)
    label: str = "all" if stage is None else stage

    rows: list[dict[str, object]] = []
    unstable: list[str] = []
    if not scored.empty:
        groups: list[tuple[str, str | None, pd.DataFrame]] = [
            (str(model_id), str(uid), part)
            for (model_id, uid), part in scored.groupby(["model_id", "unique_id"], sort=True)
        ]
        if include_overall:
            groups = [
                (str(model_id), None, part) for model_id, part in scored.groupby("model_id")
            ] + groups

        for model_id, series_id, part in groups:
            row, was_unstable = _score_group(part, quantiles=grid)
            row["model_id"] = model_id
            row["unique_id"] = series_id
            row["stage"] = label
            row["n_windows"] = int(part["window_id"].nunique())
            rows.append(row)
            if was_unstable:
                unstable.append(f"{model_id}/{series_id or 'todas'}")

    if unstable:
        warnings.warn(
            f"MAPE inestable en {len(unstable)} combinaciones de modelo y serie "
            f"({', '.join(unstable[:5])}{'...' if len(unstable) > 5 else ''}): la serie "
            "pasa por cero o cerca de cero. Ordena el leaderboard por MASE.",
            UnstableMetricWarning,
            stacklevel=2,
        )

    frame = _with_timings(pd.DataFrame(rows), result, stage=stage)
    frame = _sorted_leaderboard(frame)
    if path is not None:
        _write_parquet_atomic(frame, path)
    return frame


def _score_group(
    part: pd.DataFrame, *, quantiles: Sequence[float]
) -> tuple[dict[str, object], bool]:
    """Calcula todas las metricas de un grupo de filas de `forecasts`.

    Parameters
    ----------
    part
        Filas del grupo, con ``mase_denominator`` ya unido.
    quantiles
        Rejilla de cuantiles del run.

    Returns
    -------
    tuple
        Diccionario de metricas y un indicador de si MAPE fue inestable.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        row: dict[str, object] = {
            **point_metrics(part),
            **probabilistic_metrics(part, quantiles=quantiles),
        }

    unstable = False
    for entry in caught:
        if issubclass(entry.category, UnstableMetricWarning):
            unstable = True
        else:  # pragma: no cover  no lo emite ninguna metrica hoy
            warnings.warn_explicit(entry.message, entry.category, entry.filename, entry.lineno)
    return row, unstable


def _with_timings(
    frame: pd.DataFrame,
    result: BacktestResult,
    *,
    stage: Stage | None,
) -> pd.DataFrame:
    """Anade a cada fila el coste del modelo en las ventanas evaluadas.

    Parameters
    ----------
    frame
        Tabla de metricas por modelo y serie.
    result
        Resultado del backtest, del que sale `model_runs`.
    stage
        Etapa evaluada, para quedarse con las mismas ventanas.

    Returns
    -------
    pandas.DataFrame
        Con las columnas de coste y de recuento de ventanas por estado.
    """
    runs = result.model_runs
    if stage is not None:
        selected = result.windows.loc[result.windows["stage"] == stage, "window_id"]
        runs = runs[runs["window_id"].isin(selected)]

    rows = [
        {"model_id": str(model_id), **_timing_row(group)}
        for model_id, group in runs.groupby("model_id", sort=True)
    ]
    timings = pd.DataFrame(rows, columns=["model_id", *_TIMING_COLUMNS])

    if frame.empty:
        return pd.DataFrame(columns=[*_LEADERBOARD_HEAD, *_TIMING_COLUMNS])

    merged = frame.merge(timings, on="model_id", how="left")
    ordered = [
        *_LEADERBOARD_HEAD,
        *[c for c in merged.columns if c not in (*_LEADERBOARD_HEAD, *_TIMING_COLUMNS)],
        *_TIMING_COLUMNS,
    ]
    return merged[[column for column in ordered if column in merged.columns]]


_LEADERBOARD_HEAD: tuple[str, ...] = (
    "model_id",
    "unique_id",
    "stage",
    "n_windows",
    "n_obs",
    "mae",
    "rmse",
    "mape",
    "smape",
    "mase",
    "mase_denominator",
)
"""Columnas de cabecera del leaderboard, en el orden en que se leen."""


def _timing_row(runs: pd.DataFrame) -> dict[str, object]:
    """Resume el coste de un modelo sobre las ventanas de un run.

    Parameters
    ----------
    runs
        Filas de `model_runs` de un solo modelo.

    Returns
    -------
    dict
        Totales y medias de coste, recuento de ventanas por estado y la marca de
        zero-shot. `fit_seconds_mean` promedia **solo las ventanas en las que
        hubo ajuste**: incluir las reutilizadas diluiria el coste real de
        entrenar por la politica de refit, que es un parametro del plan y no una
        propiedad del modelo.
    """
    refits = runs[runs["refit"]]
    ok = runs[runs["status"] == "ok"]
    return {
        "fit_seconds_total": float(runs["fit_seconds"].sum()),
        "fit_seconds_mean": float(refits["fit_seconds"].mean()) if not refits.empty else np.nan,
        "predict_seconds_total": float(runs["predict_seconds"].sum()),
        "predict_seconds_mean": float(ok["predict_seconds"].mean()) if not ok.empty else np.nan,
        "n_refits": int(refits.shape[0]),
        "n_windows_ok": int((runs["status"] == "ok").sum()),
        "n_windows_failed": int((runs["status"] == "failed").sum()),
        "n_windows_skipped": int((runs["status"] == "skipped").sum()),
        "is_zero_shot": bool(runs["is_zero_shot"].any()),
    }


def _sorted_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    """Ordena la tabla: agregados por modelo primero y mejor MASE arriba.

    Parameters
    ----------
    frame
        Leaderboard sin ordenar.

    Returns
    -------
    pandas.DataFrame
        Ordenada por (agregado, serie, MASE). Los modelos sin MASE —sin
        denominador utilizable— caen al final en lugar de colarse arriba por ser
        `NaN`.
    """
    if frame.empty:
        return frame.reset_index(drop=True)

    keys = frame.assign(
        _overall=frame["unique_id"].isna(),
        _series=frame["unique_id"].fillna(""),
    ).sort_values(
        ["_overall", "_series", "mase", "model_id"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    return keys.drop(columns=["_overall", "_series"]).reset_index(drop=True)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Escribe un parquet sin dejar nunca un fichero a medias.

    Se escribe en un temporal del mismo directorio y se renombra: el renombrado
    es atomico dentro del mismo sistema de ficheros, asi que un lector solo puede
    ver la version anterior o la nueva, nunca una intermedia. Es la misma
    politica que `artifacts.writer` aplica a un run entero.

    Parameters
    ----------
    frame
        Tabla a escribir.
    path
        Destino final. Sus directorios se crean si no existen.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
