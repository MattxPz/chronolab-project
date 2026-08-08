"""Genera el subconjunto pequeno de artefactos demo que consume `chronolab.app`.

`docs/ARCHITECTURE.md` A5 dice que la app no calcula nada: solo lee artefactos
de ``reports/results/`` y dibuja. Este script es exactamente el sitio opuesto
-el que si calcula-, en el mismo papel que `scripts/run_anomaly_eval.py` y
`scripts/run_deep_analysis.py` ya cumplen para sus propios artefactos. Sin el,
las paginas Overview y Forecast no tendrian nada que leer: `leaderboard.parquet`
trae metricas ya agregadas pero ninguna prediccion cruda, y ningun artefacto
existente trae la serie en si, su descomposicion MSTL o su informe de calidad.

Deja siete artefactos nuevos en ``reports/results/``:

1. ``panel.parquet`` -la serie sintetica **contaminada** (con las mismas
   anomalias inyectadas que `anomaly_truth.parquet`): Overview, Forecast y
   Anomalias comparten la misma serie de fondo.
2. ``quality_report.parquet`` / ``quality_outliers.parquet`` -salida de
   `chronolab.data.quality`, panel de calidad de Overview.
3. ``mstl_components.parquet`` -descomposicion MSTL por serie, para el
   panel de Overview y para la descomposicion de Explicabilidad.
4. ``difficulty.parquet`` -estadisticos de dificultad por serie.
5. ``forecasts_demo.parquet`` / ``windows_demo.parquet`` -un backtest
   pequeno (siete modelos baratos, ocho ventanas) sobre el tramo final del
   panel, que es donde estan las anomalias inyectadas: los tres modelos
   estadisticos (MSTL, AutoETS) imputan los huecos que inyecta `data_gap` y
   completan las ocho ventanas; los cinco baselines no imputan
   (`ModelRequirements.handles_nan_target=False`) y alguna ventana cuyo
   entrenamiento cruza un hueco les sale ``status="failed"`` -exactamente el
   comportamiento que exige A6, no un bug de este script.
6. ``dm_matrix.parquet`` -Diebold-Mariano por pareja de esos siete modelos,
   agrupado sobre las tres series (aproximacion demo: la independencia
   temporal de Diebold-Mariano se define dentro de una serie, agrupar tres
   la debilita; `hac_lag=h-1` lo compensa parcialmente. La comparacion
   rigurosa de `leaderboard.parquet` no pasa por aqui).

La reconstruccion del panel contaminado se reusa literalmente de
`scripts.run_anomaly_eval.build_contaminated_panel`: misma semilla, mismos
parametros, para que ``s00``/``s01``/``s02`` sean identicos byte a byte a los
que ya describen `anomaly_scores.parquet` y `anomaly_truth.parquet`.

Uso: ``uv run --extra ml python scripts/build_demo_artifacts.py``
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scripts.run_anomaly_eval import build_contaminated_panel  # noqa: E402

from chronolab.data.quality import coverage_report, detect_outliers  # noqa: E402
from chronolab.evaluation.backtest import BacktestPlan, BacktestResult, backtest  # noqa: E402
from chronolab.evaluation.stats_tests import diebold_mariano  # noqa: E402
from chronolab.models.adapters.statsforecast import (  # noqa: E402
    AutoETSForecaster,
    MSTLForecaster,
)
from chronolab.models.baselines import (  # noqa: E402
    DriftForecaster,
    HistoricAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    WindowAverageForecaster,
)
from chronolab.models.protocols import Forecaster  # noqa: E402
from chronolab.panel import Panel  # noqa: E402
from chronolab.viz.plots import compute_difficulty_table, compute_mstl  # noqa: E402

RESULTS_DIR = ROOT / "reports" / "results"

H = 24
"""Horizonte del backtest demo. Igual al de `run_anomaly_eval.py` por comodidad
visual (misma escala en todas las paginas), no por necesidad tecnica."""

TRAIN_SIZE = 504
"""Tres semanas de entrenamiento deslizante: suficiente para dos ciclos
semanales (lo que exigen MSTL y AutoETS) y corto a proposito, para que el
entrenamiento de menos ventanas cruce un hueco inyectado por `data_gap`."""

N_WINDOWS = 8
HOLDOUT_WINDOWS = 3
PERIODS = (24, 168)


def _models() -> list[Forecaster]:
    """Siete modelos baratos: cinco baselines y dos estadisticos con cuantiles."""
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(season=PERIODS[0]),
        WindowAverageForecaster(window=PERIODS[0]),
        HistoricAverageForecaster(),
        DriftForecaster(),
        MSTLForecaster(h=H, season_lengths=PERIODS, calibration_windows=2),
        AutoETSForecaster(h=H, season_length=PERIODS[0], calibration_windows=2),
    ]


def build_plan() -> BacktestPlan:
    """Plan teselado de ocho ventanas, ancladas al final del panel por defecto."""
    return BacktestPlan(
        h=H,
        n_windows=N_WINDOWS,
        step_size=H,
        gap=0,
        mode="sliding",
        train_size=TRAIN_SIZE,
        holdout_windows=HOLDOUT_WINDOWS,
        refit_every=1,
        seed=20_240_807,
    )


def build_panel_artifact(panel: Panel) -> pd.DataFrame:
    """Proyecta el panel a las columnas de `chronolab.artifacts.schemas.panel_schema`."""
    columns = ["unique_id", "ds", *panel.spec.value_columns]
    return panel.df[columns].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def build_quality_artifacts(panel: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Informe de calidad y filas atipicas, sobre el panel contaminado.

    El generador sintetico produce rejilla completa y sin duplicados por
    construccion, asi que la trama "cruda" y la "alineada" que pide
    `coverage_report` son la misma: los huecos que si aparecen son los que
    `data_gap` inyecto, no un artefacto de la fuente.
    """
    report = coverage_report(panel.df, panel.df, value_column=panel.spec.target)
    outliers = detect_outliers(panel.df, value_column=panel.spec.target)
    columns = ["unique_id", "ds", panel.spec.target, "robust_z"]
    outliers = outliers[columns].rename(columns={panel.spec.target: "y"})
    return report, outliers.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def build_mstl_artifact(panel: Panel) -> pd.DataFrame:
    """Descomposicion MSTL por serie, en formato largo."""
    parts: list[pd.DataFrame] = []
    for uid in panel.ids():
        series = panel.df.loc[panel.df["unique_id"] == uid].set_index("ds")[panel.spec.target]
        components = compute_mstl(series, periods=PERIODS)
        components = components.reset_index(names="ds")
        components.insert(0, "unique_id", str(uid))
        parts.append(components)
    return pd.concat(parts, ignore_index=True)


def build_difficulty_artifact(panel: Panel) -> pd.DataFrame:
    """Estadisticos de dificultad, una fila por serie."""
    series_map = {
        str(uid): panel.df.loc[panel.df["unique_id"] == uid].set_index("ds")[panel.spec.target]
        for uid in panel.ids()
    }
    return compute_difficulty_table(series_map, periods=PERIODS)


def run_demo_backtest(panel: Panel) -> BacktestResult:
    """Backtest de los siete modelos demo sobre el tramo final del panel."""
    plan = build_plan()
    models = _models()
    started = time.perf_counter()
    with warnings.catch_warnings():
        # AutoARIMA (dentro de MSTL) avisa de convergencia en algunas ventanas
        # cortas; el ajuste que devuelve sigue siendo utilizable.
        warnings.simplefilter("ignore", UserWarning)
        result = backtest(panel, models, plan)
    elapsed = time.perf_counter() - started

    runs = result.model_runs
    print(f"backtest demo: {elapsed:.1f}s, {len(runs)} filas modelo x ventana")
    print(runs.groupby(["model_id", "status"]).size().unstack(fill_value=0).to_string())
    return result


def build_dm_matrix(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano por pareja de modelos, sobre el error absoluto pooled entre series.

    Parameters
    ----------
    forecasts
        Tabla `forecasts` del backtest demo, con ``model_id``, ``unique_id``,
        ``ds``, ``y`` y ``y_hat``.

    Returns
    -------
    pandas.DataFrame
        Una fila por pareja **ordenada** ``(model_a, model_b)``, con el
        esquema de `chronolab.artifacts.schemas.dm_matrix_schema`.
    """
    scored = forecasts.dropna(subset=["y", "y_hat"]).copy()
    scored["abs_error"] = (scored["y"] - scored["y_hat"]).abs()
    wide = scored.pivot_table(index=["unique_id", "ds"], columns="model_id", values="abs_error")
    wide = wide.sort_index(level="ds")

    model_ids = sorted(wide.columns)
    rows: list[dict[str, object]] = []
    for a in model_ids:
        for b in model_ids:
            if a == b:
                continue
            paired = wide[[a, b]].dropna()
            if len(paired) < 10:
                continue
            result = diebold_mariano(
                paired[a].to_numpy(), paired[b].to_numpy(), hac_lag=H - 1, hln=True
            )
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "stat": result.stat,
                    "p_value": result.p_value,
                    "n_obs": result.n_obs,
                    "mean_difference": result.mean_difference,
                    "degenerate": result.degenerate,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Genera y persiste los siete artefactos demo."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("reconstruyendo el panel contaminado (misma semilla que run_anomaly_eval.py)...")
    panel, _truth, _anomaly_plan = build_contaminated_panel()

    print("panel + calidad + MSTL + dificultad...")
    build_panel_artifact(panel).to_parquet(RESULTS_DIR / "panel.parquet", index=False)
    quality_report, quality_outliers = build_quality_artifacts(panel)
    quality_report.to_parquet(RESULTS_DIR / "quality_report.parquet", index=False)
    quality_outliers.to_parquet(RESULTS_DIR / "quality_outliers.parquet", index=False)
    build_mstl_artifact(panel).to_parquet(RESULTS_DIR / "mstl_components.parquet", index=False)
    build_difficulty_artifact(panel).to_parquet(RESULTS_DIR / "difficulty.parquet", index=False)

    result = run_demo_backtest(panel)
    windows = result.windows
    forecasts = result.forecasts.merge(windows[["window_id", "stage"]], on="window_id", how="left")
    forecasts.to_parquet(RESULTS_DIR / "forecasts_demo.parquet", index=False)
    windows.to_parquet(RESULTS_DIR / "windows_demo.parquet", index=False)

    print("Diebold-Mariano por pareja de modelos...")
    dm_matrix = build_dm_matrix(forecasts)
    dm_matrix.to_parquet(RESULTS_DIR / "dm_matrix.parquet", index=False)

    written = [
        "panel.parquet",
        "quality_report.parquet",
        "quality_outliers.parquet",
        "mstl_components.parquet",
        "difficulty.parquet",
        "forecasts_demo.parquet",
        "windows_demo.parquet",
        "dm_matrix.parquet",
    ]
    print("\nescritos:")
    for name in written:
        path = RESULTS_DIR / name
        print(f"  {path} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
