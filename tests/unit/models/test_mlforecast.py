"""LightGBMForecaster y XGBoostForecaster: protocolo, estrategias y explicabilidad.

Los lags/ventanas/diferencias de `TargetFeatureConfig` se acortan a proposito
(`_SMALL_TARGET`, `_SMALL_THERMAL`) frente a los canonicos del proyecto
(`chronolab.features.builders.DEFAULT_TARGET_FEATURES`, con lags hasta 336):
lo que se comprueba aqui es que el envoltorio cumple el protocolo, traduce
columnas correctamente y filtra las exogenas manuales por adelanto, no la
calidad estadistica de un ajuste con la configuracion completa.

`mlforecast`, `lightgbm` y `xgboost` viven en el extra `ml` (D20), no en el
nucleo: `pytest.importorskip` a nivel de modulo salta todo el fichero con
limpieza en el job `quality` de CI, que hace `uv sync` a secas.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlforecast")
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")

from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.features.builders import TargetFeatureConfig, ThermalFeatureConfig
from chronolab.models.adapters.mlforecast import LightGBMForecaster, XGBoostForecaster
from chronolab.models.protocols import FittedForecaster, Forecaster
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId, ModelId

pytestmark = pytest.mark.slow

H = 6
N_HOURS = 24 * 12

_SMALL_TARGET = TargetFeatureConfig(
    lags=(1, 2, 3, 24),
    roll_windows=(3,),
    roll_stats=("mean", "std"),
    diff_lags=(1,),
    pct_change_lags=(3,),
)
_SMALL_THERMAL = ThermalFeatureConfig(lags=(1, 2))


def _panel(n_series: int = 2, *, futr_temp: bool = True) -> Panel:
    rng = np.random.default_rng(4)
    index = pd.date_range("2023-01-02", periods=N_HOURS, freq="h")
    temp = 12 + 8 * np.sin(2 * np.pi * (index.hour - 4) / 24) + rng.normal(0, 0.5, N_HOURS)
    parts = []
    for i in range(n_series):
        level = 50 + 5 * np.sin(2 * np.pi * np.arange(N_HOURS) / 24) + i * 3
        y = level + 0.3 * np.abs(temp - 16) + rng.normal(0, 0.5, N_HOURS)
        parts.append(pd.DataFrame({"unique_id": f"s{i}", "ds": index, "y": y, "temp_c": temp}))
    frame = pd.concat(parts, ignore_index=True)
    role = {"futr_exog": ("temp_c",)} if futr_temp else {"hist_exog": ("temp_c",)}
    spec = PanelSpec(
        dataset_id=DatasetId("mini"),
        freq="h",
        seasonalities=(24,),
        tz_display="Europe/Madrid",
        **role,  # type: ignore[arg-type]
    )
    return Panel(df=frame, spec=spec)


def _models() -> list[Forecaster]:
    return [
        LightGBMForecaster(
            strategy="recursive",
            target_features=_SMALL_TARGET,
            thermal=_SMALL_THERMAL,
            params={"n_estimators": 20, "num_leaves": 7},
            model_id=ModelId("lightgbm_recursive"),
        ),
        LightGBMForecaster(
            strategy="direct",
            target_features=_SMALL_TARGET,
            thermal=_SMALL_THERMAL,
            params={"n_estimators": 20, "num_leaves": 7},
            model_id=ModelId("lightgbm_direct"),
        ),
        XGBoostForecaster(
            strategy="recursive",
            target_features=_SMALL_TARGET,
            thermal=_SMALL_THERMAL,
            params={"n_estimators": 20, "max_depth": 3},
            model_id=ModelId("xgboost_recursive"),
        ),
        XGBoostForecaster(
            strategy="direct",
            target_features=_SMALL_TARGET,
            thermal=_SMALL_THERMAL,
            params={"n_estimators": 20, "max_depth": 3},
            model_id=ModelId("xgboost_direct"),
        ),
    ]


def _model_ids() -> list[str]:
    return ["lightgbm_recursive", "lightgbm_direct", "xgboost_recursive", "xgboost_direct"]


@pytest.fixture(scope="module")
def panel() -> Panel:
    return _panel()


class TestConformidadConElProtocolo:
    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_satisface_forecaster_y_fitted_forecaster(
        self, model: Forecaster, panel: Panel
    ) -> None:
        assert isinstance(model, Forecaster)
        fitted = model.fit(panel, h=H)
        assert isinstance(fitted, FittedForecaster)
        assert fitted.cutoff == panel.last_ds
        assert fitted.h == H
        assert fitted.n_params is None
        assert fitted.fit_seconds >= 0.0

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_ninguno_necesita_exogenas_futuras(self, model: Forecaster) -> None:
        # Calendario y termicas se reconstruyen enteras a partir de `ds`
        # (docstring del modulo): no hace falta el valor de ninguna exogena.
        assert model.requires.needs_futr_exog is False

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_predict_devuelve_exactamente_h_filas_por_serie(
        self, model: Forecaster, panel: Panel
    ) -> None:
        fitted = model.fit(panel, h=H)
        prediction = fitted.predict()

        assert len(prediction) == H * len(panel.ids())
        assert set(prediction["unique_id"]) == set(panel.ids())
        assert (prediction["ds"] > panel.last_ds).all()

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_los_cuantiles_calibrados_estan_ordenados(
        self, model: Forecaster, panel: Panel
    ) -> None:
        fitted = model.fit(panel, h=H)
        prediction = fitted.predict()

        columns = [c for c in ("q_0250", "q_1000", "q_5000", "q_9000", "q_9750") if c in prediction]
        assert columns == ["q_0250", "q_1000", "q_5000", "q_9000", "q_9750"]
        values = prediction[columns].to_numpy()
        assert (np.diff(values, axis=1) >= 0).all()

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_la_mediana_es_el_pronostico_puntual(self, model: Forecaster, panel: Panel) -> None:
        fitted = model.fit(panel, h=H)
        prediction = fitted.predict()
        pd.testing.assert_series_equal(prediction["q_5000"], prediction["y_hat"], check_names=False)

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_supports_recursive_solo_en_la_variante_recursiva(self, model: Forecaster) -> None:
        is_recursive = "recursive" in str(model.model_id)
        assert model.requires.supports_recursive is is_recursive


class TestEstrategiaRecursivaVsDirecta:
    def test_direct_ajusta_h_submodelos_y_recursive_uno(self, panel: Panel) -> None:
        recursive = LightGBMForecaster(
            strategy="recursive",
            target_features=_SMALL_TARGET,
            thermal=_SMALL_THERMAL,
            params={"n_estimators": 10},
        ).fit(panel, h=H)
        direct = LightGBMForecaster(
            strategy="direct",
            target_features=_SMALL_TARGET,
            thermal=_SMALL_THERMAL,
            params={"n_estimators": 10},
        ).fit(panel, h=H)

        assert len(recursive._estimators()) == 1
        assert len(direct._estimators()) == H


class TestExplicabilidad:
    def test_feature_importance_cubre_todas_las_features_y_esta_ordenada(
        self, panel: Panel
    ) -> None:
        fitted = LightGBMForecaster(
            target_features=_SMALL_TARGET, thermal=_SMALL_THERMAL, params={"n_estimators": 10}
        ).fit(panel, h=H)

        importance = fitted.feature_importance()

        assert set(importance["feature"]) == set(fitted.mlf.ts.features_order_)
        assert (importance["importance"] >= 0).all()
        assert importance["importance"].is_monotonic_decreasing

    def test_shap_values_cubre_todas_las_features_y_es_no_negativo(self, panel: Panel) -> None:
        fitted = LightGBMForecaster(
            target_features=_SMALL_TARGET, thermal=_SMALL_THERMAL, params={"n_estimators": 10}
        ).fit(panel, h=H)

        shap_frame = fitted.shap_values(sample_size=20, seed=0)

        assert set(shap_frame["feature"]) == set(fitted.mlf.ts.features_order_)
        assert (shap_frame["mean_abs_shap"] >= 0).all()
        assert shap_frame["mean_abs_shap"].is_monotonic_decreasing

    def test_shap_sin_shap_instalado_lanza_import_error_claro(
        self, monkeypatch: pytest.MonkeyPatch, panel: Panel
    ) -> None:
        fitted = LightGBMForecaster(
            target_features=_SMALL_TARGET, thermal=None, params={"n_estimators": 5}
        ).fit(panel, h=H)
        for name in list(sys.modules):
            if name == "shap" or name.startswith("shap."):
                monkeypatch.setitem(sys.modules, name, None)

        with pytest.raises(ImportError, match="extra 'ml'"):
            fitted.shap_values()


class TestMinContext:
    def test_min_context_domina_el_lag_mas_largo(self) -> None:
        # El lag mas largo es 24 (objetivo), frente a roll (3+1=4), diff
        # (1+1=2), pct_change (3+1=4) y termicas (2). Domina 24, +1 de margen
        # tras el dropna que aplica mlforecast al preprocesar.
        model = LightGBMForecaster(target_features=_SMALL_TARGET, thermal=_SMALL_THERMAL)
        assert model.requires.min_context == 25

    def test_sin_termicas_el_minimo_no_cambia_si_no_dominan(self) -> None:
        model = LightGBMForecaster(target_features=_SMALL_TARGET, thermal=None)
        assert model.requires.min_context == 25


class TestValidacionDeParametros:
    def test_rechaza_estrategia_no_admitida(self) -> None:
        with pytest.raises(ValueError, match="strategy debe ser"):
            LightGBMForecaster(strategy="weird")  # type: ignore[arg-type]

    def test_rechaza_niveles_fuera_de_rango(self) -> None:
        with pytest.raises(ValueError, match="nivel de intervalo fuera de"):
            XGBoostForecaster(levels=(80, 101))

    def test_rechaza_menos_de_dos_ventanas_de_calibracion(self) -> None:
        with pytest.raises(ValueError, match="calibration_windows debe ser >= 2"):
            LightGBMForecaster(calibration_windows=1)


class TestFiltradoDeExogenasPorAdelanto:
    def test_termicas_historicas_se_excluyen_mas_alla_de_su_ultimo_retardo(self) -> None:
        hist_panel = _panel(n_series=1, futr_temp=False)  # temp_c hist_exog: max_lead(temp)=0
        thermal = ThermalFeatureConfig(lags=(1, 2))

        within = LightGBMForecaster(
            strategy="direct",
            target_features=_SMALL_TARGET,
            thermal=thermal,
            params={"n_estimators": 5},
        ).fit(hist_panel, h=2)
        beyond = LightGBMForecaster(
            strategy="direct",
            target_features=_SMALL_TARGET,
            thermal=thermal,
            params={"n_estimators": 5},
        ).fit(hist_panel, h=5)

        assert within.thermal_names  # h=2 <= max(lags)=2: sobreviven
        assert not beyond.thermal_names  # h=5 > max(lags)=2: ninguna sobrevive
        assert len(beyond.predict()) == 5

    def test_el_filtro_no_depende_del_rol_de_la_temperatura(self) -> None:
        # A diferencia del algebra general de `features.roles` (donde una
        # temperatura futr_exog haria UNBOUNDED cualquier retardo), este
        # adaptador nunca lee el `FutrFrame`: el filtro es "k >= h" a secas,
        # igual con futr_exog que con hist_exog, porque en ningun caso hay de
        # donde sacar el valor real mas alla de la historia reconstruida.
        thermal = ThermalFeatureConfig(lags=(1, 2))
        futr_fitted = LightGBMForecaster(
            strategy="direct",
            target_features=_SMALL_TARGET,
            thermal=thermal,
            params={"n_estimators": 5},
        ).fit(_panel(n_series=1, futr_temp=True), h=5)
        hist_fitted = LightGBMForecaster(
            strategy="direct",
            target_features=_SMALL_TARGET,
            thermal=thermal,
            params={"n_estimators": 5},
        ).fit(_panel(n_series=1, futr_temp=False), h=5)

        assert futr_fitted.thermal_names == ()
        assert hist_fitted.thermal_names == ()

    def test_las_termicas_reconstruidas_no_tienen_nan_en_el_tramo_futuro(self) -> None:
        thermal = ThermalFeatureConfig(lags=(1, 2))
        fitted = LightGBMForecaster(
            strategy="direct",
            target_features=_SMALL_TARGET,
            thermal=thermal,
            params={"n_estimators": 5},
        ).fit(_panel(n_series=1, futr_temp=True), h=2)

        future_grid = pd.DataFrame(
            {
                "unique_id": ["s0"] * 2,
                "ds": pd.date_range(fitted.cutoff + pd.Timedelta(hours=1), periods=2, freq="h"),
            }
        )
        x_df = fitted._future_regressors(future_grid)

        assert x_df is not None
        assert not x_df.isna().any().any()


class TestUsoIntervalsDesactivado:
    def test_sin_intervalos_solo_queda_la_mediana(self, panel: Panel) -> None:
        model = LightGBMForecaster(
            target_features=_SMALL_TARGET,
            thermal=_SMALL_THERMAL,
            use_intervals=False,
            params={"n_estimators": 10},
        )
        fitted = model.fit(panel, h=H)
        prediction = fitted.predict()

        assert list(prediction.columns) == ["unique_id", "ds", "y_hat", "q_5000"]
        pd.testing.assert_series_equal(prediction["q_5000"], prediction["y_hat"], check_names=False)


class TestImportPerezoso:
    def test_sin_mlforecast_instalado_el_mensaje_dice_que_instalar(
        self, monkeypatch: pytest.MonkeyPatch, panel: Panel
    ) -> None:
        for name in list(sys.modules):
            if name == "mlforecast" or name.startswith("mlforecast."):
                monkeypatch.setitem(sys.modules, name, None)

        model = LightGBMForecaster(target_features=_SMALL_TARGET, thermal=None)
        with pytest.raises(ImportError, match="extra 'ml'"):
            model.fit(panel, h=H)

    def test_sin_lightgbm_instalado_el_mensaje_dice_que_instalar(
        self, monkeypatch: pytest.MonkeyPatch, panel: Panel
    ) -> None:
        for name in list(sys.modules):
            if name == "lightgbm" or name.startswith("lightgbm."):
                monkeypatch.setitem(sys.modules, name, None)

        model = LightGBMForecaster(target_features=_SMALL_TARGET, thermal=None)
        with pytest.raises(ImportError, match="extra 'ml'"):
            model.fit(panel, h=H)


class TestIntegracionConElMotor:
    def test_backtest_recursivo_y_directo_no_producen_fuga(self, panel: Panel) -> None:
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=1)
        result = backtest(panel, _models(), plan)

        assert (result.model_runs["status"] == "ok").all()
        assert (result.forecasts["ds"] > result.forecasts["cutoff"]).all()
