"""`tune()`: Optuna limitado por construccion a las ventanas `dev` del backtest.

Se tunea `SeasonalNaiveForecaster`/`WindowAverageForecaster`
(`chronolab.models.baselines`) en vez de LightGBM o XGBoost a proposito: lo
que se comprueba aqui es la mecanica del tuning —el recorte del panel, que
Optuna nunca ve el holdout, el presupuesto de trials, la metrica
configurable, que un trial sin observaciones no tira el estudio—, no la
calidad de un modelo de arboles. Eso mantiene la suite rapida y evita que este
fichero dependa de `lightgbm`/`xgboost` ademas de `optuna`.

`optuna` vive en el extra `ml` (D20), no en el nucleo: `pytest.importorskip`
a nivel de modulo salta todo el fichero con limpieza en el job `quality` de
CI, que hace `uv sync` a secas.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")

from chronolab.evaluation.backtest import BacktestPlan
from chronolab.evaluation.tuning import TuningResult, dev_only_panel, tune
from chronolab.models.baselines import SeasonalNaiveForecaster, WindowAverageForecaster
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId, ModelId

SEASON = 4
H = 2
N_HOURS = 60


def _panel(y: np.ndarray | None = None) -> Panel:
    index = pd.date_range("2023-01-02", periods=N_HOURS, freq="h")
    if y is None:
        y = 50.0 + 5.0 * np.sin(2 * np.pi * np.arange(N_HOURS) / SEASON)
    frame = pd.DataFrame({"unique_id": "s0", "ds": index, "y": y})
    spec = PanelSpec(dataset_id=DatasetId("mini"), freq="h", seasonalities=(SEASON,))
    return Panel(df=frame, spec=spec)


def _plan(*, holdout_windows: int = 1) -> BacktestPlan:
    return BacktestPlan(h=H, n_windows=4, step_size=H, holdout_windows=holdout_windows)


class TestDevOnlyPanel:
    def test_recorta_el_panel_exactamente_al_final_de_la_ultima_ventana_dev(self) -> None:
        panel = _panel()
        plan = _plan()
        windows = plan.splitter().split(panel)
        last_dev = max(w.last_pred for w in windows if w.stage == "dev")

        trimmed, dev_plan = dev_only_panel(panel, plan)

        assert trimmed.last_ds == last_dev
        assert trimmed.last_ds < panel.last_ds  # el holdout desaparece de verdad
        assert dev_plan.n_windows == 3
        assert dev_plan.holdout_windows == 0

    def test_las_ventanas_dev_reconstruidas_coinciden_con_las_del_plan_original(self) -> None:
        panel = _panel()
        plan = _plan()
        original_dev = tuple(w for w in plan.splitter().split(panel) if w.stage == "dev")

        trimmed, dev_plan = dev_only_panel(panel, plan)
        reconstructed = dev_plan.splitter().split(trimmed)

        assert [w.cutoff for w in reconstructed] == [w.cutoff for w in original_dev]
        assert [w.train_start for w in reconstructed] == [w.train_start for w in original_dev]
        assert all(w.stage == "dev" for w in reconstructed)

    def test_sin_ventanas_dev_lanza_un_error_claro(self) -> None:
        panel = _panel()
        plan = _plan(holdout_windows=4)
        with pytest.raises(ValueError, match="ninguna ventana 'dev'"):
            dev_only_panel(panel, plan)


class TestTune:
    def test_encuentra_la_estacion_correcta(self) -> None:
        panel = _panel()
        plan = _plan()

        def build_model(trial: Any) -> SeasonalNaiveForecaster:
            season = trial.suggest_categorical("season", [3, SEASON, 5])
            return SeasonalNaiveForecaster(season=season, model_id=ModelId("candidate"))

        result = tune(panel, build_model, plan, n_trials=6, seed=0)

        assert isinstance(result, TuningResult)
        assert result.study.best_params["season"] == SEASON
        assert result.n_dev_windows == 3
        assert result.dev_plan.holdout_windows == 0

    def test_el_tuning_no_ve_un_holdout_con_un_patron_distinto(self) -> None:
        # El tramo que sera holdout se distorsiona a proposito: si `tune()`
        # colase esas observaciones en el objetivo, elegiria una estacion
        # distinta de la que de verdad explica las ventanas dev.
        y = 50.0 + 5.0 * np.sin(2 * np.pi * np.arange(N_HOURS) / SEASON)
        y[-2 * H :] = y[-2 * H :][::-1] * 5.0 + 1000.0
        panel = _panel(y)
        plan = _plan()

        def build_model(trial: Any) -> SeasonalNaiveForecaster:
            season = trial.suggest_categorical("season", [3, SEASON, 5])
            return SeasonalNaiveForecaster(season=season, model_id=ModelId("candidate"))

        result = tune(panel, build_model, plan, n_trials=6, seed=0)

        assert result.study.best_params["season"] == SEASON

    def test_metrica_configurable(self) -> None:
        panel = _panel()
        plan = _plan()

        def build_model(trial: Any) -> WindowAverageForecaster:
            window = trial.suggest_int("window", 2, 8)
            return WindowAverageForecaster(window=window, model_id=ModelId("candidate"))

        result = tune(panel, build_model, plan, n_trials=5, metric="mae", seed=0)

        assert "window" in result.study.best_params

    def test_un_trial_sin_observaciones_no_tira_el_estudio(self) -> None:
        panel = _panel()
        plan = _plan()

        def build_model(trial: Any) -> WindowAverageForecaster:
            # window=1000 excede min_context en todas las ventanas: se saltan
            # todas, `forecasts` queda vacio para ese trial.
            window = trial.suggest_categorical("window", [3, 1000])
            return WindowAverageForecaster(window=window, model_id=ModelId("candidate"))

        result = tune(panel, build_model, plan, n_trials=4, seed=0)

        assert len(result.study.trials) == 4
        assert math.isfinite(result.study.best_value)
        assert result.study.best_params["window"] == 3

    def test_presupuesto_de_trials_configurable(self) -> None:
        panel = _panel()
        plan = _plan()

        def build_model(trial: Any) -> SeasonalNaiveForecaster:
            season = trial.suggest_int("season", 2, 8)
            return SeasonalNaiveForecaster(season=season, model_id=ModelId("candidate"))

        result = tune(panel, build_model, plan, n_trials=3, seed=0)

        assert len(result.study.trials) == 3


class TestImportPerezoso:
    def test_sin_optuna_instalado_el_mensaje_dice_que_instalar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in list(sys.modules):
            if name == "optuna" or name.startswith("optuna."):
                monkeypatch.setitem(sys.modules, name, None)

        panel = _panel()
        plan = _plan()
        with pytest.raises(ImportError, match="extra 'ml'"):
            tune(panel, lambda trial: SeasonalNaiveForecaster(), plan, n_trials=1)
