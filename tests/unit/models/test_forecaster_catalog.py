"""Un `Forecaster` de cada familia registrada, sobre una unica serie diminuta.

Cada adaptador ya tiene su propio fichero de conformidad (`test_baselines.py`,
`test_statistical.py`, `test_mlforecast.py`, `test_neuralforecast.py`,
`test_prophet.py`, `test_torch_lstm.py`), con fixtures y aserciones especificas
de su familia. Lo que ninguno de ellos comprueba es la propiedad transversal:
que **todas** las familias, puestas una al lado de otra sobre el mismo panel
pequeno, satisfacen el mismo contrato minimo y producen un backtest limpio sin
que nadie tenga que acordarse de anadir el modelo nuevo a media docena de
ficheros distintos. Este es el equivalente, para modelos, del catalogo
transversal de detectores en
`tests/unit/anomaly/test_detector_catalog.py`.

No hay una `chronolab.models.registry` poblada todavia (D21 documenta el
diseno, pero el modulo es solo el docstring): el "registro" de este test es la
lista `CATALOG` de abajo, mantenida a mano. Cuando se implemente el registro de
verdad, este fichero es el sitio natural para sustituir la lista por una
iteracion sobre el.

`chronos` queda fuera a proposito: es zero-shot y su unico camino offline pasa
por sustituir `_load_pipeline` por un doble de prueba, una pieza de
infraestructura que ya vive en `test_chronos.py` y que reproducirla aqui solo
anadiria codigo sin anadir cobertura.

Las familias que ajustan de verdad (statsforecast, mlforecast, neuralforecast,
torch, prophet) se marcan `slow`, igual que sus ficheros dedicados: `make
test-fast` solo ejercita los baselines, y `make test` ejercita el catalogo
completo.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from chronolab.data.futr import RealizedFutrProvider
from chronolab.errors import PerfectForesightWarning
from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.models.baselines import (
    DriftForecaster,
    HistoricAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    WindowAverageForecaster,
)
from chronolab.models.protocols import FittedForecaster, Forecaster
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId

H = 6
N_HOURS = 24 * 21
"""Tres semanas horarias: por encima del piso de 14 dias de Prophet y de sobra
para el contexto de 24 pasos de neuralforecast/torch, pero muy por debajo de
cualquier panel real del proyecto."""


def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in modules)


def _needs_extra(*modules: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not _installed(*modules), reason=f"requiere {', '.join(modules)}, no instalado"
    )


def _panel(n_series: int = 2) -> Panel:
    """Panel horario diminuto compartido por todo el catalogo.

    Con estacionalidad diaria y semanal, tendencia y una exogena futura
    (`temp_c`), es suficiente para que ninguna familia se quede corta de
    contexto ni de columnas declaradas.
    """
    rng = np.random.default_rng(99)
    index = pd.date_range("2023-01-02", periods=N_HOURS, freq="h")
    temp = 12.0 + 8.0 * np.sin(2 * np.pi * (index.hour - 4) / 24) + rng.normal(0, 0.5, N_HOURS)
    parts = []
    for i in range(n_series):
        level = (
            100.0
            + 10.0 * i
            + 8.0 * np.sin(2 * np.pi * np.arange(N_HOURS) / 24)
            + 3.0 * np.sin(2 * np.pi * np.arange(N_HOURS) / 168)
        )
        y = level + 0.3 * np.abs(temp - 16.0) + rng.normal(0, 0.6, N_HOURS)
        parts.append(pd.DataFrame({"unique_id": f"s{i}", "ds": index, "y": y, "temp_c": temp}))
    frame = pd.concat(parts, ignore_index=True)
    spec = PanelSpec(
        dataset_id=DatasetId("catalog_mini"),
        freq="h",
        seasonalities=(24, 168),
        futr_exog=("temp_c",),
        tz_display="Europe/Madrid",
    )
    return Panel(df=frame, spec=spec)


PANEL = _panel()
PLAN = BacktestPlan(h=H, n_windows=1)


def _naive() -> Forecaster:
    return NaiveForecaster()


def _seasonal_naive() -> Forecaster:
    return SeasonalNaiveForecaster(season=24)


def _window_average() -> Forecaster:
    return WindowAverageForecaster(window=24)


def _historic_average() -> Forecaster:
    return HistoricAverageForecaster()


def _drift() -> Forecaster:
    return DriftForecaster()


def _auto_arima() -> Forecaster:
    from chronolab.models.adapters.statsforecast import AutoARIMAForecaster

    return AutoARIMAForecaster(h=H, season_length=24, max_p=1, max_q=1, max_P=0, max_Q=0)


def _auto_ets() -> Forecaster:
    from chronolab.models.adapters.statsforecast import AutoETSForecaster

    return AutoETSForecaster(h=H, season_length=24)


def _auto_theta() -> Forecaster:
    from chronolab.models.adapters.statsforecast import AutoThetaForecaster

    return AutoThetaForecaster(h=H, season_length=24)


def _mstl() -> Forecaster:
    from chronolab.models.adapters.statsforecast import MSTLForecaster

    return MSTLForecaster(h=H, season_lengths=(24, 168), trend_max_p=1, trend_max_q=1)


def _lightgbm() -> Forecaster:
    from chronolab.features.builders import TargetFeatureConfig, ThermalFeatureConfig
    from chronolab.models.adapters.mlforecast import LightGBMForecaster

    return LightGBMForecaster(
        strategy="recursive",
        target_features=TargetFeatureConfig(
            lags=(1, 2, 24), roll_windows=(3,), roll_stats=("mean",), diff_lags=(1,)
        ),
        thermal=ThermalFeatureConfig(lags=(1, 2)),
        params={"n_estimators": 20, "num_leaves": 7},
    )


def _xgboost() -> Forecaster:
    from chronolab.features.builders import TargetFeatureConfig, ThermalFeatureConfig
    from chronolab.models.adapters.mlforecast import XGBoostForecaster

    return XGBoostForecaster(
        strategy="direct",
        target_features=TargetFeatureConfig(
            lags=(1, 2, 24), roll_windows=(3,), roll_stats=("mean",), diff_lags=(1,)
        ),
        thermal=ThermalFeatureConfig(lags=(1, 2)),
        params={"n_estimators": 20, "max_depth": 3},
    )


def _nhits() -> Forecaster:
    from chronolab.models.adapters.neuralforecast import NHITSForecaster, quiet_lightning

    quiet_lightning()
    return NHITSForecaster(
        input_size=24,
        mlp_units=((16, 16),) * 3,
        max_steps=8,
        val_check_steps=4,
        early_stop_patience_steps=2,
    )


def _torch_lstm() -> Forecaster:
    from chronolab.models.adapters.torch_lstm import LSTMForecaster
    from chronolab.models.torch.trainer import TrainConfig

    return LSTMForecaster(
        input_size=24,
        hidden_size=8,
        num_layers=1,
        config=TrainConfig(max_epochs=3, batch_size=64, patience=2, val_fraction=0.2),
    )


def _prophet() -> Forecaster:
    from chronolab.models.adapters.prophet import ProphetForecaster

    return ProphetForecaster(uncertainty_samples=100)


CATALOG: list[pytest.param] = [
    pytest.param(_naive, id="naive"),
    pytest.param(_seasonal_naive, id="seasonal_naive"),
    pytest.param(_window_average, id="window_average"),
    pytest.param(_historic_average, id="historic_average"),
    pytest.param(_drift, id="drift"),
    pytest.param(
        _auto_arima, id="auto_arima", marks=(pytest.mark.slow, _needs_extra("statsforecast"))
    ),
    pytest.param(_auto_ets, id="auto_ets", marks=(pytest.mark.slow, _needs_extra("statsforecast"))),
    pytest.param(
        _auto_theta, id="auto_theta", marks=(pytest.mark.slow, _needs_extra("statsforecast"))
    ),
    pytest.param(_mstl, id="mstl", marks=(pytest.mark.slow, _needs_extra("statsforecast"))),
    pytest.param(
        _lightgbm,
        id="lightgbm",
        marks=(pytest.mark.slow, _needs_extra("mlforecast", "lightgbm")),
    ),
    pytest.param(
        _xgboost,
        id="xgboost",
        marks=(pytest.mark.slow, _needs_extra("mlforecast", "xgboost")),
    ),
    pytest.param(
        _nhits, id="nhits", marks=(pytest.mark.slow, _needs_extra("neuralforecast", "torch"))
    ),
    pytest.param(_torch_lstm, id="torch_lstm", marks=(pytest.mark.slow, _needs_extra("torch"))),
    pytest.param(_prophet, id="prophet", marks=(pytest.mark.slow, _needs_extra("prophet"))),
]


@pytest.fixture(params=CATALOG)
def factory(request: pytest.FixtureRequest) -> Callable[[], Forecaster]:
    return request.param  # type: ignore[no-any-return]


class TestConformidadEstructural:
    """`fit` sobre el panel completo, sin pasar por el motor: solo el protocolo."""

    def test_es_un_forecaster(self, factory: Callable[[], Forecaster]) -> None:
        assert isinstance(factory(), Forecaster)

    def test_fit_devuelve_un_fittedforecaster_con_el_cutoff_del_panel(
        self, factory: Callable[[], Forecaster]
    ) -> None:
        fitted = factory().fit(PANEL, h=H)
        assert isinstance(fitted, FittedForecaster)
        assert fitted.cutoff == PANEL.last_ds
        assert fitted.h == H
        assert fitted.fit_seconds >= 0.0


class TestBacktestLimpio:
    """El mismo modelo, dentro del motor real, produce exactamente `h` filas por serie."""

    def test_un_backtest_de_una_ventana_termina_en_ok(
        self, factory: Callable[[], Forecaster]
    ) -> None:
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=PANEL)

        result = backtest(PANEL, [factory()], PLAN, futr=provider)

        assert (result.model_runs["status"] == "ok").all(), result.model_runs[
            ["window_id", "status", "error"]
        ].to_dict("records")
        assert len(result.forecasts) == H * len(PANEL.ids())
        assert (result.forecasts["ds"] > result.forecasts["cutoff"]).all()
