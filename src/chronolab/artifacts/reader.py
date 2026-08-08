"""Unica ruta de lectura de artefactos. Es la API que consume la app.

Tambien es el unico constructor de `ScoringFrame`, lo que garantiza que un
detector jamas reciba predicciones dentro de muestra (fuga L9).

Dos variantes conviven aqui:

- **En memoria**: `scoring_frame`, que construye un `ScoringFrame` a partir de
  un `BacktestResult` sin pasar por parquet. La usan los scripts de
  evaluacion (p. ej. `scripts/run_anomaly_eval.py`), justo despues de
  `chronolab.evaluation.backtest.backtest`.
- **En disco**: las funciones `load_*`, que leen las tablas ya persistidas en
  ``reports/results/`` (por los scripts de evaluacion o por
  `scripts/build_demo_artifacts.py`) y las validan contra
  `chronolab.artifacts.schemas` antes de devolverlas. Es la unica via por la
  que `chronolab.app` toca un DataFrame (docs/ARCHITECTURE.md §2.1): la app no
  llama a `pandas.read_parquet` directamente en ningun sitio.

Ninguna funcion de este modulo usa `st.cache_data`: el cacheo entre
`rerun`s de Streamlit es responsabilidad de `chronolab.app.components.state`,
que envuelve estas funciones. Mantenerlo fuera de aqui es lo que permite
probar el lector con `pytest` sin arrancar Streamlit.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandera.pandas as pa

from chronolab.anomaly.protocols import ScoringFrame
from chronolab.artifacts import schemas
from chronolab.config import results_dir as _default_results_dir
from chronolab.errors import ArtifactNotFound, PredictionContractError, WindowValidationError
from chronolab.types import ModelId, Stage

if TYPE_CHECKING:  # pragma: no cover
    # Import diferido: en tiempo de ejecucion aqui solo se leen atributos del
    # resultado, asi que `artifacts` no depende de `evaluation`. Es el mismo
    # patron que usa el motor con `FutrProvider`.
    from chronolab.evaluation.backtest import BacktestResult

__all__ = [
    "ARTIFACT_FILES",
    "SCORING_FRAME_COLUMNS",
    "available_artifacts",
    "load_anomaly_results",
    "load_anomaly_scores",
    "load_anomaly_truth",
    "load_difficulty",
    "load_dm_matrix",
    "load_forecasts",
    "load_leaderboard",
    "load_mstl_components",
    "load_panel",
    "load_quality_outliers",
    "load_quality_report",
    "load_tft_interpretability",
    "load_windows",
    "scoring_frame",
]

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


# --------------------------------------------------------------------------- #
# Lectura desde disco: la variante que consume `chronolab.app`
# --------------------------------------------------------------------------- #

ARTIFACT_FILES: dict[str, str] = {
    "leaderboard": "leaderboard.parquet",
    "anomaly_scores": "anomaly_scores.parquet",
    "anomaly_results": "anomaly_results.parquet",
    "anomaly_truth": "anomaly_truth.parquet",
    "tft_interpretability": "tft_interpretability.parquet",
    "panel": "panel.parquet",
    "quality_report": "quality_report.parquet",
    "quality_outliers": "quality_outliers.parquet",
    "mstl_components": "mstl_components.parquet",
    "difficulty": "difficulty.parquet",
    "forecasts": "forecasts_demo.parquet",
    "windows": "windows_demo.parquet",
    "dm_matrix": "dm_matrix.parquet",
}
"""Nombre logico -> nombre de fichero en ``reports/results/``.

Nombres, no `Path` completos: quien resuelve el directorio es cada `load_*`
(o `available_artifacts`), a partir de `chronolab.config.results_dir()` salvo
que se pase `results_dir` explicito -lo que necesitan los tests, que no deben
tocar el `reports/results/` real del repositorio.
"""


def available_artifacts(results_dir: Path | None = None) -> dict[str, bool]:
    """Que artefactos de `ARTIFACT_FILES` existen, sin leerlos.

    Pensada para que la app decida que paginas o paneles puede dibujar sin
    intentar cargar y capturar la excepcion: una tabla en `st.sidebar` con el
    estado de cada artefacto es mas util que un intento fallido por pagina.

    Parameters
    ----------
    results_dir
        Directorio a inspeccionar. Por defecto, `chronolab.config.results_dir`.

    Returns
    -------
    dict[str, bool]
        ``nombre_logico -> existe``.
    """
    base = results_dir if results_dir is not None else _default_results_dir()
    return {name: (base / filename).exists() for name, filename in ARTIFACT_FILES.items()}


def _read_table(
    name: str, schema: pa.DataFrameSchema, *, results_dir: Path | None = None
) -> pd.DataFrame:
    """Lee y valida una tabla de `ARTIFACT_FILES` por su nombre logico.

    Parameters
    ----------
    name
        Clave de `ARTIFACT_FILES`.
    schema
        Esquema pandera contra el que validar (`.validate(frame, lazy=True)`).
    results_dir
        Directorio a leer. Por defecto, `chronolab.config.results_dir`.

    Returns
    -------
    pandas.DataFrame
        Tabla validada.

    Raises
    ------
    ArtifactNotFound
        Si el fichero no existe. El mensaje nombra el artefacto y el comando
        que lo genera, para que quien lo lea sepa que ejecutar en vez de leer
        una traza.
    """
    base = results_dir if results_dir is not None else _default_results_dir()
    path = base / ARTIFACT_FILES[name]
    if not path.exists():
        raise ArtifactNotFound(
            f"falta el artefacto '{name}' ({path}). En modo demo se genera con "
            f"`uv run --extra ml python scripts/build_demo_artifacts.py` "
            f"(o `run_anomaly_eval.py` para las tablas de anomalias)."
        )
    frame = pd.read_parquet(path)
    validated: pd.DataFrame = schema.validate(frame, lazy=True)
    return validated


def load_leaderboard(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Tabla de metricas por modelo, serie y etapa. Fuente de la pagina Leaderboard."""
    return _read_table("leaderboard", schemas.leaderboard_schema(), results_dir=results_dir)


def load_anomaly_scores(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Scores crudos de los detectores sobre el holdout. Fuente de la pagina Anomalias."""
    return _read_table("anomaly_scores", schemas.anomaly_scores_schema(), results_dir=results_dir)


def load_anomaly_results(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Metricas de deteccion por detector y tipo de anomalia."""
    return _read_table("anomaly_results", schemas.anomaly_results_schema(), results_dir=results_dir)


def load_anomaly_truth(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Ground truth disperso de la inyeccion sintetica de anomalias."""
    return _read_table("anomaly_truth", schemas.anomaly_truth_schema(), results_dir=results_dir)


def load_tft_interpretability(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Pesos de atencion del TFT. Fuente parcial de la pagina Explicabilidad."""
    return _read_table(
        "tft_interpretability", schemas.tft_interpretability_schema(), results_dir=results_dir
    )


def load_panel(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Panel demo: la serie cruda (con las anomalias inyectadas) que ve el resto de la app."""
    return _read_table("panel", schemas.panel_schema(), results_dir=results_dir)


def load_quality_report(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Informe de calidad por serie (`chronolab.data.quality.coverage_report`)."""
    return _read_table("quality_report", schemas.quality_report_schema(), results_dir=results_dir)


def load_quality_outliers(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Filas marcadas como atipicas (`chronolab.data.quality.detect_outliers`)."""
    return _read_table(
        "quality_outliers", schemas.quality_outliers_schema(), results_dir=results_dir
    )


def load_mstl_components(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Descomposicion MSTL precalculada por serie."""
    return _read_table("mstl_components", schemas.mstl_components_schema(), results_dir=results_dir)


def load_difficulty(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Estadisticos de dificultad de cada serie."""
    return _read_table("difficulty", schemas.difficulty_schema(), results_dir=results_dir)


def load_forecasts(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Predicciones crudas del backtest demo. Fuente de la pagina Forecast."""
    return _read_table("forecasts", schemas.forecasts_schema(), results_dir=results_dir)


def load_windows(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Ventanas efectivas del backtest demo."""
    return _read_table("windows", schemas.windows_schema(), results_dir=results_dir)


def load_dm_matrix(*, results_dir: Path | None = None) -> pd.DataFrame:
    """Contrastes de Diebold-Mariano por pareja de modelos del backtest demo."""
    return _read_table("dm_matrix", schemas.dm_matrix_schema(), results_dir=results_dir)
