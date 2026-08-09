"""Test de humo de extremo a extremo: todo el pipeline sobre 200 puntos sinteticos.

No comprueba ninguna barrera en detalle -para eso esta `tests/leakage/` y el
resto de la suite unitaria- sino que la tuberia completa **encaja**: un panel
sintetico entra por un extremo, sale un leaderboard persistido y una tabla de
eventos de anomalias por el otro, sin excepciones, en un tiempo acotado. Es la
red que atrapa una integracion rota que ningun test unitario ve, porque cada
uno prueba su propio modulo con sus propios dobles.

Deliberadamente **no** se marca `slow`: vive en el presupuesto de `make
test-fast` y del job `quality` de CI, que es donde hace falta que una
regresion de integracion se note en cada PR, no solo en el smoke nocturno con
los extras `ml`/`deep` instalados.

El presupuesto de 60 segundos es una asercion del propio test, no una
convencion de CI: si la tuberia se vuelve mas lenta, este test lo dice con una
cifra en el mensaje de fallo en vez de con un timeout opaco del runner.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronolab.anomaly.conformal import ConformalDetector
from chronolab.anomaly.events import aggregate_events
from chronolab.artifacts.reader import scoring_frame
from chronolab.evaluation.aggregate import build_leaderboard, score_forecasts
from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.models.baselines import NaiveForecaster, SeasonalNaiveForecaster
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId, ModelId

BUDGET_SECONDS = 60.0
N_POINTS = 200
"""Puntos totales del panel: una sola serie horaria de poco mas de ocho dias."""

H = 6
N_WINDOWS = 8
HOLDOUT_WINDOWS = 2
SEASONAL_MODEL = ModelId("seasonal_naive")


def _synthetic_panel() -> Panel:
    """Panel horario de `N_POINTS` puntos, con estacionalidad diaria y ruido."""
    rng = np.random.default_rng(2024)
    index = pd.date_range("2023-01-02", periods=N_POINTS, freq="h")
    daily = 10.0 * np.sin(2 * np.pi * index.hour.to_numpy() / 24)
    trend = 0.05 * np.arange(N_POINTS)
    y = 100.0 + daily + trend + rng.normal(0, 1.0, N_POINTS)
    frame = pd.DataFrame({"unique_id": "s0", "ds": index, "y": y})
    spec = PanelSpec(dataset_id=DatasetId("smoke_e2e"), freq="h", seasonalities=(24,))
    return Panel(df=frame, spec=spec)


@pytest.fixture(scope="module")
def started() -> float:
    return time.perf_counter()


def test_el_pipeline_completo_corre_en_menos_de_60_segundos(started: float, tmp_path: Path) -> None:
    panel = _synthetic_panel()
    assert len(panel.df) == N_POINTS

    # 1. Backtesting: dos baselines, origen rodante, con holdout reservado.
    plan = BacktestPlan(h=H, n_windows=N_WINDOWS, step_size=H, holdout_windows=HOLDOUT_WINDOWS)
    result = backtest(panel, [NaiveForecaster(), SeasonalNaiveForecaster(season=24)], plan)
    assert (result.model_runs["status"] == "ok").all(), result.model_runs.to_dict("records")
    assert not result.forecasts.empty

    # 2. Metricas y leaderboard, persistido de verdad (ruta atomica real).
    leaderboard_path = tmp_path / "leaderboard.parquet"
    leaderboard = build_leaderboard(result, panel, stage="holdout", path=leaderboard_path)
    assert leaderboard_path.exists()
    assert not leaderboard.empty
    assert leaderboard["mase"].notna().any()

    scored = score_forecasts(result, panel, stage="holdout")
    assert (scored["mase_denominator"] > 0).all()

    # 3. Deteccion de anomalias sobre los residuos del naive estacional:
    #    calibracion en desarrollo, puntuacion en holdout.
    calib = scoring_frame(result, model_id=SEASONAL_MODEL, stage="dev")
    holdout = scoring_frame(result, model_id=SEASONAL_MODEL, stage="holdout")
    detector = ConformalDetector(
        base_model_id=SEASONAL_MODEL, hour_bins=1, min_calib=10, gamma=0.02, pool_size=50
    )
    fitted = detector.fit(calib)
    scores = fitted.score(holdout)
    assert len(scores) == len(holdout.df)

    # 4. Agregacion en eventos, el ultimo eslabon antes de dibujarlos en la app.
    events = aggregate_events(scores, detector_id=detector.detector_id, alpha=0.1)
    assert {"unique_id", "start_ds", "end_ds"}.issubset(events.columns)

    elapsed = time.perf_counter() - started
    assert elapsed < BUDGET_SECONDS, (
        f"el pipeline completo tardo {elapsed:.1f}s, por encima del presupuesto "
        f"de {BUDGET_SECONDS:.0f}s"
    )
