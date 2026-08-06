"""`ChronosForecaster`: adaptador zero-shot, sin descargar pesos reales.

`chronos-forecasting` vive en el extra `deep` (D20): `pytest.importorskip` a
nivel de modulo salta el fichero entero con limpieza en el job `quality` de CI,
que hace `uv sync` a secas.

Los tests **no descargan pesos**: `_load_pipeline` se sustituye por un
pipeline de juguete con la misma interfaz (`predict_quantiles`, `inner_model`),
asi que la suite corre offline y en milisegundos. Lo que se comprueba es que
el adaptador cumple el protocolo, traduce el contexto y los cuantiles
correctamente, y respeta el mismo contrato de horizonte que el resto de
adaptadores — no la calidad de las predicciones de Chronos, para lo que esta
el backtest del hito. El fallback sin red se prueba aparte, restaurando el
`_load_pipeline` real con `_require_chronos` roto a proposito.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("chronos")

import chronolab.models.adapters.chronos as chronos_adapter
from chronolab.data.futr import RealizedFutrProvider
from chronolab.errors import FoundationModelUnavailable, PerfectForesightWarning
from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.models.adapters.chronos import ChronosForecaster
from chronolab.models.protocols import QUANTILES, FittedForecaster, Forecaster
from chronolab.panel import FutrFrame, Panel, PanelSpec
from chronolab.types import DatasetId, Vintage

_REAL_LOAD_PIPELINE = chronos_adapter._load_pipeline
"""Referencia capturada antes de que el fixture autouse la sustituya."""

H = 6
N_HOURS = 24 * 10
QUANTILE_COLUMNS = [f"q_{round(q * 10000):04d}" for q in QUANTILES]


class _FakeInnerModel:
    """Sustituto minimo del modelo interno: solo hace falta contar parametros."""

    def __init__(self) -> None:
        import torch

        self._params = [torch.zeros(3, 4), torch.zeros(5)]

    def parameters(self) -> Any:
        return iter(self._params)

    def to(self, device: str) -> _FakeInnerModel:
        return self

    def eval(self) -> _FakeInnerModel:
        return self


class _FakePipeline:
    """Pipeline de juguete: cada cuantil es la base mas un desplazamiento fijo.

    La base de cada paso es ``ultimo_punto_de_contexto + paso``, y el
    desplazamiento de cada cuantil es ``(nivel - 0.5) * 10``, de modo que el
    nivel 0.5 reproduce la base exacta: facil de comprobar a mano y distinto
    por serie y por paso.
    """

    def __init__(self) -> None:
        self.inner_model = _FakeInnerModel()

    def predict_quantiles(
        self, inputs: Any, prediction_length: int, quantile_levels: list[float], **kwargs: Any
    ) -> tuple[Any, Any]:
        import torch

        last = torch.stack([series[-1] for series in inputs])  # (batch,)
        steps = torch.arange(1, prediction_length + 1, dtype=torch.float32)
        base = last[:, None] + steps[None, :]  # (batch, h)
        offsets = torch.tensor([(q - 0.5) * 10.0 for q in quantile_levels], dtype=torch.float32)
        quantiles = base[:, :, None] + offsets[None, None, :]
        return quantiles, base


def _panel(n_series: int = 2) -> Panel:
    rng = np.random.default_rng(7)
    index = pd.date_range("2023-01-02", periods=N_HOURS, freq="h")
    parts = [
        pd.DataFrame(
            {
                "unique_id": f"s{i}",
                "ds": index,
                "y": 100.0 + 10.0 * i + rng.normal(0, 0.1, N_HOURS),
            }
        )
        for i in range(n_series)
    ]
    spec = PanelSpec(dataset_id=DatasetId("mini"), freq="h", seasonalities=(24,))
    return Panel(df=pd.concat(parts, ignore_index=True), spec=spec)


def _futr_frame(panel: Panel, grid: pd.DatetimeIndex) -> FutrFrame:
    frame = pd.concat(
        [pd.DataFrame({"unique_id": uid, "ds": grid}) for uid in panel.ids()], ignore_index=True
    )
    return FutrFrame(df=frame, window=None, vintage=Vintage.REALIZED)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _fake_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sustituye la carga de pesos por un pipeline de juguete en toda la suite."""
    chronos_adapter._load_pipeline.cache_clear()
    monkeypatch.setattr(
        chronos_adapter, "_load_pipeline", lambda pretrained, device: _FakePipeline()
    )


@pytest.fixture
def panel() -> Panel:
    return _panel()


class TestConfiguracion:
    def test_model_id_por_defecto_coincide_con_el_pretrained_por_defecto(self) -> None:
        model = ChronosForecaster()
        assert model.model_id == "chronos-bolt-small"
        assert model.pretrained_model_name_or_path == "amazon/chronos-bolt-small"

    def test_es_zero_shot_sin_exogenas_y_de_coste_libre(self) -> None:
        requires = ChronosForecaster().requires
        assert requires.is_zero_shot is True
        assert requires.needs_futr_exog is False
        assert requires.uses_hist_exog is False
        assert requires.uses_static_exog is False
        assert requires.supports_quantiles is True
        assert requires.refit_cost == "free"

    @pytest.mark.parametrize("kwargs", [{"context_length": 0}, {"min_context": 0}])
    def test_rechaza_configuraciones_invalidas(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            ChronosForecaster(**kwargs)  # type: ignore[arg-type]

    def test_context_length_none_es_valido(self) -> None:
        assert ChronosForecaster(context_length=None).context_length is None


class TestAdaptador:
    def test_satisface_forecaster_y_fitted_forecaster(self, panel: Panel) -> None:
        model = ChronosForecaster()
        assert isinstance(model, Forecaster)

        fitted = model.fit(panel, h=H)

        assert isinstance(fitted, FittedForecaster)
        assert fitted.cutoff == panel.last_ds
        assert fitted.h == H

    def test_fit_no_entrena_es_practicamente_instantaneo(self, panel: Panel) -> None:
        # No hay descarga ni carga de pesos dentro de fit (esta cacheada aparte):
        # solo se recorta el contexto de cada serie.
        assert ChronosForecaster().fit(panel, h=H).fit_seconds < 0.5

    def test_n_params_es_un_numero_real_del_pipeline_cargado(self, panel: Panel) -> None:
        fitted = ChronosForecaster().fit(panel, h=H)
        assert fitted.n_params == 3 * 4 + 5

    def test_el_contexto_se_recorta_a_context_length(self, panel: Panel) -> None:
        fitted = ChronosForecaster(context_length=24).fit(panel, h=H)
        assert all(len(values) == 24 for values in fitted.context.values())

    def test_sin_context_length_no_se_recorta(self, panel: Panel) -> None:
        fitted = ChronosForecaster(context_length=None).fit(panel, h=H)
        assert all(len(values) == N_HOURS for values in fitted.context.values())

    def test_una_serie_toda_nan_falla_al_ajustar(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": "s0",
                "ds": pd.date_range("2023-01-02", periods=48, freq="h"),
                "y": np.nan,
            }
        )
        spec = PanelSpec(dataset_id=DatasetId("mini"), freq="h", seasonalities=(24,))
        vacio = Panel(df=frame, spec=spec)
        with pytest.raises(ValueError, match="ninguna observacion valida"):
            ChronosForecaster().fit(vacio, h=H)

    def test_los_huecos_se_rellenan_solo_con_el_pasado(self) -> None:
        valores = [10.0, np.nan, 30.0, np.nan, 50.0]
        frame = pd.DataFrame(
            {
                "unique_id": "s0",
                "ds": pd.date_range("2023-01-02", periods=len(valores), freq="h"),
                "y": valores,
            }
        )
        spec = PanelSpec(dataset_id=DatasetId("mini"), freq="h", seasonalities=(2,))
        con_huecos = Panel(df=frame, spec=spec)

        fitted = ChronosForecaster().fit(con_huecos, h=H)

        np.testing.assert_allclose(fitted.context["s0"], [10.0, 10.0, 30.0, 30.0, 50.0])

    def test_la_prediccion_tiene_el_esquema_del_protocolo(self, panel: Panel) -> None:
        fitted = ChronosForecaster().fit(panel, h=H)

        prediction = fitted.predict(quantiles=QUANTILES)

        assert set(prediction.columns) == {"unique_id", "ds", "y_hat", *QUANTILE_COLUMNS}
        assert len(prediction) == len(panel.ids()) * H
        assert (prediction["ds"] > fitted.cutoff).all()
        assert set(prediction["unique_id"]) == {str(uid) for uid in panel.ids()}

    def test_la_mediana_es_el_punto_y_los_cuantiles_salen_ordenados(self, panel: Panel) -> None:
        fitted = ChronosForecaster().fit(panel, h=H)

        prediction = fitted.predict(quantiles=QUANTILES)

        np.testing.assert_allclose(prediction["y_hat"], prediction["q_5000"])
        values = prediction[QUANTILE_COLUMNS].to_numpy()
        assert (np.diff(values, axis=1) >= 0).all()

    def test_los_cuantiles_pedidos_son_reales_nunca_nan(self, panel: Panel) -> None:
        # A diferencia de neuralforecast (rejilla fija entrenada), aqui
        # cualquier nivel se calcula: no hay "cuantil no entrenado".
        fitted = ChronosForecaster().fit(panel, h=H)
        prediction = fitted.predict(quantiles=(0.05, 0.37, 0.63, 0.95))
        assert not prediction[["q_0500", "q_3700", "q_6300", "q_9500"]].isna().any().any()

    def test_pide_siempre_la_mediana_aunque_no_se_pida_en_quantiles(self, panel: Panel) -> None:
        fitted = ChronosForecaster().fit(panel, h=H)

        prediction = fitted.predict(quantiles=(0.1, 0.9))

        assert set(prediction.columns) == {"unique_id", "ds", "y_hat", "q_1000", "q_9000"}
        for uid, group in prediction.sort_values("ds").groupby("unique_id"):
            base = fitted.context[str(uid)][-1] + np.arange(1, H + 1, dtype=np.float32)
            np.testing.assert_allclose(group["y_hat"].to_numpy(), base, atol=1e-4)
            np.testing.assert_allclose(group["q_1000"].to_numpy(), base - 4.0, atol=1e-4)
            np.testing.assert_allclose(group["q_9000"].to_numpy(), base + 4.0, atol=1e-4)

    def test_usa_los_instantes_del_futrframe_si_llega(self, panel: Panel) -> None:
        fitted = ChronosForecaster().fit(panel, h=H)
        grid = pd.date_range(fitted.cutoff, periods=H + 1, freq="h")[1:]

        prediction = fitted.predict(_futr_frame(panel, grid))

        assert set(prediction["ds"]) == set(grid)

    def test_reutilizar_el_ajuste_fuera_de_1_h_falla_con_un_mensaje_util(
        self, panel: Panel
    ) -> None:
        fitted = ChronosForecaster().fit(panel, h=H)
        # Pide los pasos H+1..2H en vez de 1..H: la ventana siguiente si se
        # reutilizase este ajuste sin reajustar.
        grid = pd.date_range(fitted.cutoff, periods=2 * H + 1, freq="h")[H + 1 :]

        with pytest.raises(ValueError, match="refit_every=1"):
            fitted.predict(_futr_frame(panel, grid))


class TestIntegracionConElMotor:
    def test_un_backtest_sin_futrprovider_no_produce_fuga(self, panel: Panel) -> None:
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=1)

        result = backtest(panel, [ChronosForecaster()], plan)

        assert (result.model_runs["status"] == "ok").all()
        assert (result.forecasts["ds"] > result.forecasts["cutoff"]).all()
        assert not result.forecasts["q_5000"].isna().any()
        assert result.model_runs["is_zero_shot"].all()

    def test_ignora_un_futrprovider_pensado_para_otros_modelos(self, panel: Panel) -> None:
        # El motor pasa el FutrFrame a todos los modelos del run aunque
        # Chronos no declare needs_futr_exog: no debe fallar por recibirlo.
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=1)

        result = backtest(panel, [ChronosForecaster()], plan, futr=provider)

        assert (result.model_runs["status"] == "ok").all()

    def test_reutilizar_el_ajuste_con_refit_every_mayor_que_uno_falla_ruidosamente(
        self, panel: Panel
    ) -> None:
        # Con un FutrProvider en el run, `_assert_within_horizon` detecta la
        # reutilizacion y falla con un mensaje util antes de que el motor la
        # vea como fuga generica. Sin proveedor, el fallback (`cutoff + 1..h`
        # del propio ajuste) no tiene con que compararse y el motor la detecta
        # de todos modos, pero como `CutoffViolation`: es leakage real —la
        # ventana 1 predeciria instantes anteriores a su propio cutoff— y
        # aborta el run entero en lugar de marcar solo ese modelo.
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=2)

        result = backtest(panel, [ChronosForecaster()], plan, futr=provider)

        assert result.model_runs["status"].tolist() == ["ok", "failed"]
        assert "refit_every=1" in result.model_runs["error"].dropna().iloc[0]

    def test_reutilizar_sin_futrprovider_se_ve_como_fuga_y_aborta_el_run(
        self, panel: Panel
    ) -> None:
        from chronolab.errors import CutoffViolation

        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=2)

        with pytest.raises(CutoffViolation):
            backtest(panel, [ChronosForecaster()], plan)

    def test_una_ventana_demasiado_corta_se_salta(self, panel: Panel) -> None:
        plan = BacktestPlan(h=H, n_windows=1)
        # Ningun tramo de entrenamiento de este panel llega a tantos pasos.
        model = ChronosForecaster(min_context=N_HOURS * 10)

        result = backtest(panel, [model], plan)

        assert (result.model_runs["status"] == "skipped").all()


class TestFallbackSinRed:
    def test_sin_red_ni_cache_falla_con_un_mensaje_util(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _BrokenPipelineClass:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> None:
                raise OSError("simulated: no network, no local cache")

        def _broken_require() -> tuple[Any, Any]:
            return _BrokenPipelineClass, None

        monkeypatch.setattr(chronos_adapter, "_require_chronos", _broken_require)

        with pytest.raises(FoundationModelUnavailable, match="sin red"):
            _REAL_LOAD_PIPELINE.__wrapped__("no-existe/modelo-inventado", "cpu")

    def test_un_error_ajeno_al_de_red_no_se_disfraza(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _bad_config_require() -> tuple[Any, Any]:
            class _NotAChronosPipeline:
                @staticmethod
                def from_pretrained(*args: object, **kwargs: object) -> None:
                    raise ValueError("Not a Chronos config file")

            return _NotAChronosPipeline, None

        monkeypatch.setattr(chronos_adapter, "_require_chronos", _bad_config_require)

        with pytest.raises(ValueError, match="Not a Chronos config file"):
            _REAL_LOAD_PIPELINE.__wrapped__("no-es-chronos/lo-que-sea", "cpu")
