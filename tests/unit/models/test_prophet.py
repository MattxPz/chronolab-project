"""ProphetForecaster: conformidad de protocolo, regresores y festivos.

Ajustar Prophet de verdad cuesta CPU real (varios fits por serie en el
fichero): todo el modulo se marca `slow`, igual que
`tests/unit/models/test_statistical.py`.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from chronolab.data.futr import RealizedFutrProvider
from chronolab.errors import MissingFutrExog, PerfectForesightWarning
from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.models.adapters.prophet import ProphetForecaster
from chronolab.models.protocols import FittedForecaster, Forecaster
from chronolab.panel import FutrFrame, Panel, PanelSpec
from chronolab.types import DatasetId

pytestmark = pytest.mark.slow

H = 24
N_HOURS = 24 * 20  # 20 dias: por encima del piso de _MIN_TRAIN_HOURS (14 dias)


def _panel(n_series: int = 2) -> Panel:
    rng = np.random.default_rng(2)
    index = pd.date_range("2023-01-02", periods=N_HOURS, freq="h")
    temp = 12 + 8 * np.sin(2 * np.pi * (index.hour - 4) / 24) + rng.normal(0, 0.5, N_HOURS)
    level = 100 + 10 * np.sin(2 * np.pi * np.arange(N_HOURS) / 24) + 0.4 * np.abs(temp - 16)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": index,
                    "y": level + i * 5 + rng.normal(0, 1, N_HOURS),
                    "temp_c": temp,
                }
            )
            for i in range(n_series)
        ],
        ignore_index=True,
    )
    spec = PanelSpec(
        dataset_id=DatasetId("mini"), freq="h", seasonalities=(24, 168), futr_exog=("temp_c",)
    )
    return Panel(df=frame, spec=spec)


def _future_frame(panel: Panel, *, h: int = H) -> FutrFrame:
    rng = np.random.default_rng(3)
    grid = pd.date_range(panel.last_ds, periods=h + 1, freq="h")[1:]
    temp = 12 + 8 * np.sin(2 * np.pi * (grid.hour - 4) / 24) + rng.normal(0, 0.5, h)
    frame = pd.concat(
        [pd.DataFrame({"unique_id": uid, "ds": grid, "temp_c": temp}) for uid in panel.ids()],
        ignore_index=True,
    )
    from chronolab.types import Vintage

    return FutrFrame(df=frame, window=None, vintage=Vintage.REALIZED)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def panel() -> Panel:
    return _panel()


@pytest.fixture(scope="module")
def fitted(panel: Panel):
    model = ProphetForecaster(uncertainty_samples=100)
    return model.fit(panel, h=H)


class TestConformidadConElProtocolo:
    def test_satisface_forecaster_y_fitted_forecaster(self, panel: Panel, fitted) -> None:
        assert isinstance(ProphetForecaster(), Forecaster)
        assert isinstance(fitted, FittedForecaster)
        assert fitted.cutoff == panel.last_ds
        assert fitted.h == H
        assert fitted.n_params is None
        assert fitted.fit_seconds >= 0.0

    def test_por_defecto_necesita_exogenas_futuras(self) -> None:
        assert ProphetForecaster().requires.needs_futr_exog is True

    def test_sin_regresores_no_necesita_exogenas_futuras(self) -> None:
        assert ProphetForecaster(regressors=()).requires.needs_futr_exog is False

    def test_rechaza_muestras_de_incertidumbre_no_positivas(self) -> None:
        with pytest.raises(ValueError, match="uncertainty_samples debe ser >= 1"):
            ProphetForecaster(uncertainty_samples=0)


class TestPrediccionConRegresor:
    def test_predict_devuelve_exactamente_h_filas_por_serie(self, panel: Panel, fitted) -> None:
        prediction = fitted.predict(_future_frame(panel))

        assert len(prediction) == H * len(panel.ids())
        assert set(prediction["unique_id"]) == set(panel.ids())
        assert (prediction["ds"] > panel.last_ds).all()

    def test_todas_las_columnas_de_cuantil_estan_presentes_y_ordenadas(
        self, panel: Panel, fitted
    ) -> None:
        prediction = fitted.predict(_future_frame(panel))
        columns = ["q_0250", "q_1000", "q_2500", "q_5000", "q_7500", "q_9000", "q_9750"]

        assert all(column in prediction.columns for column in columns)
        values = prediction[columns].to_numpy()
        assert (np.diff(values, axis=1) >= 0).all()

    def test_sin_futrframe_hace_falta_el_regresor(self, fitted) -> None:
        with pytest.raises(MissingFutrExog, match="regresores"):
            fitted.predict(None)

    def test_un_futrframe_sin_el_regresor_registrado_lanza_un_error_claro(
        self, panel: Panel, fitted
    ) -> None:
        from chronolab.types import Vintage

        incompleto = FutrFrame(
            df=pd.DataFrame(
                {
                    "unique_id": list(panel.ids()),
                    "ds": [panel.last_ds + pd.Timedelta(hours=1)] * len(panel.ids()),
                }
            ),
            window=None,  # type: ignore[arg-type]
            vintage=Vintage.REALIZED,
        )
        with pytest.raises(ValueError, match="faltan los regresores"):
            fitted.predict(incompleto)


class TestSinRegresores:
    def test_predict_usa_cutoff_mas_freq_como_reserva(self) -> None:
        panel = _panel(n_series=1)
        model = ProphetForecaster(regressors=(), uncertainty_samples=50)
        fitted = model.fit(panel, h=H)

        prediction = fitted.predict()

        esperado = pd.date_range(panel.last_ds, periods=H + 1, freq="h")[1:]
        assert prediction["ds"].tolist() == list(esperado)


class TestFestivos:
    def test_country_holidays_none_no_rompe_el_ajuste(self) -> None:
        panel = _panel(n_series=1)
        model = ProphetForecaster(country_holidays=None, uncertainty_samples=50)
        fitted = model.fit(panel, h=H)
        prediction = fitted.predict(_future_frame(panel))
        assert len(prediction) == H


class TestImportPerezoso:
    def test_sin_prophet_instalado_el_mensaje_dice_que_instalar(
        self, monkeypatch: pytest.MonkeyPatch, panel: Panel
    ) -> None:
        for name in list(sys.modules):
            if name == "prophet" or name.startswith("prophet."):
                monkeypatch.setitem(sys.modules, name, None)

        model = ProphetForecaster(uncertainty_samples=50)
        with pytest.raises(ImportError, match="extra 'ml'"):
            model.fit(panel, h=H)


class TestIntegracionConElMotor:
    def test_un_backtest_con_futrprovider_no_produce_fuga(self, panel: Panel) -> None:
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=1)
        model = ProphetForecaster(uncertainty_samples=50)

        result = backtest(panel, [model], plan, futr=provider)

        assert (result.model_runs["status"] == "ok").all()
        assert (result.forecasts["ds"] > result.forecasts["cutoff"]).all()
        assert not result.forecasts["q_5000"].isna().any()

    def test_sin_futrprovider_el_run_aborta(self, panel: Panel) -> None:
        plan = BacktestPlan(h=H, n_windows=1)
        with pytest.raises(MissingFutrExog):
            backtest(panel, [ProphetForecaster()], plan)
