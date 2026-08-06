"""NHITS, TFT y PatchTST: protocolo, cuantiles nativos e interpretabilidad del TFT.

Redes minusculas y `max_steps` de dos cifras a proposito: lo que se comprueba
es que el envoltorio cumple el contrato, que la traduccion de nombres de
columna de neuralforecast a la convencion `q_<int>` del proyecto es correcta, y
que los pesos de seleccion de variables y de atencion del TFT salen en el
formato que espera la tabla `explanations`. La calidad del ajuste es cosa del
backtest del hito.

`neuralforecast` vive en el extra `deep` (D20), no en el nucleo:
`pytest.importorskip` a nivel de modulo salta el fichero entero con limpieza en
el job `quality` de CI, que hace `uv sync` a secas.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neuralforecast")

from chronolab.data.futr import RealizedFutrProvider
from chronolab.errors import PerfectForesightWarning
from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.models.adapters.neuralforecast import (
    NHITSForecaster,
    PatchTSTForecaster,
    TFTForecaster,
    quiet_lightning,
)
from chronolab.models.protocols import QUANTILES, FittedForecaster, Forecaster
from chronolab.panel import FutrFrame, Panel, PanelSpec
from chronolab.types import DatasetId, Vintage

pytestmark = pytest.mark.slow

quiet_lightning()

H = 6
INPUT_SIZE = 24
N_HOURS = 24 * 14
QUANTILE_COLUMNS = [f"q_{round(q * 10000):04d}" for q in QUANTILES]

_BUDGET = {"max_steps": 8, "val_check_steps": 4, "early_stop_patience_steps": 2}


def _panel(n_series: int = 2) -> Panel:
    rng = np.random.default_rng(13)
    index = pd.date_range("2023-01-02", periods=N_HOURS, freq="h")
    temp = 12 + 8 * np.sin(2 * np.pi * (index.hour - 4) / 24) + rng.normal(0, 0.5, N_HOURS)
    parts = []
    for i in range(n_series):
        level = 50 + 5 * np.sin(2 * np.pi * np.arange(N_HOURS) / 24) + 10 * i
        y = level + 0.3 * np.abs(temp - 16) + rng.normal(0, 0.5, N_HOURS)
        parts.append(pd.DataFrame({"unique_id": f"s{i}", "ds": index, "y": y, "temp_c": temp}))
    spec = PanelSpec(
        dataset_id=DatasetId("mini"),
        freq="h",
        seasonalities=(24,),
        futr_exog=("temp_c",),
        tz_display="Europe/Madrid",
    )
    return Panel(df=pd.concat(parts, ignore_index=True), spec=spec)


def _futr_frame(panel: Panel, h: int) -> FutrFrame:
    grid = pd.date_range(panel.last_ds, periods=h + 1, freq="h")[1:]
    rng = np.random.default_rng(5)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": grid,
                    "temp_c": 12
                    + 8 * np.sin(2 * np.pi * (grid.hour - 4) / 24)
                    + rng.normal(0, 0.5, h),
                }
            )
            for uid in panel.ids()
        ],
        ignore_index=True,
    )
    return FutrFrame(df=frame, window=None, vintage=Vintage.REALIZED)  # type: ignore[arg-type]


def _models() -> list[Forecaster]:
    return [
        NHITSForecaster(input_size=INPUT_SIZE, mlp_units=((16, 16),) * 3, **_BUDGET),
        TFTForecaster(input_size=INPUT_SIZE, hidden_size=8, n_head=2, **_BUDGET),
        PatchTSTForecaster(
            input_size=INPUT_SIZE,
            hidden_size=16,
            linear_hidden_size=16,
            encoder_layers=1,
            n_heads=2,
            patch_len=8,
            stride=4,
            use_futr_exog=False,
            **_BUDGET,
        ),
    ]


def _model_ids() -> list[str]:
    return ["nhits", "tft", "patchtst"]


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
        assert fitted.fit_seconds > 0.0

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_n_params_es_un_numero_real(self, model: Forecaster, panel: Panel) -> None:
        fitted = model.fit(panel, h=H)
        assert isinstance(fitted.n_params, int)
        assert fitted.n_params > 0

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_predict_devuelve_exactamente_h_filas_por_serie(
        self, model: Forecaster, panel: Panel
    ) -> None:
        fitted = model.fit(panel, h=H)
        prediction = fitted.predict(_futr_frame(panel, H))

        assert len(prediction) == H * len(panel.ids())
        assert set(prediction["unique_id"]) == set(panel.ids())
        assert (prediction["ds"] > panel.last_ds).all()

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_todos_los_cuantiles_de_la_rejilla_se_estiman(
        self, model: Forecaster, panel: Panel
    ) -> None:
        # La regresion que este test protege: `MQLoss` guarda su rejilla en
        # float32, asi que emparejar por igualdad exacta dejaria todas estas
        # columnas en NaN sin que nada lo dijese.
        fitted = model.fit(panel, h=H)
        prediction = fitted.predict(_futr_frame(panel, H))

        assert not prediction[QUANTILE_COLUMNS].isna().any().any()

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_la_mediana_es_el_pronostico_puntual(self, model: Forecaster, panel: Panel) -> None:
        fitted = model.fit(panel, h=H)
        prediction = fitted.predict(_futr_frame(panel, H))
        np.testing.assert_allclose(prediction["y_hat"], prediction["q_5000"], atol=1e-9)

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_declara_un_unico_ajuste_por_run(self, model: Forecaster) -> None:
        assert model.requires.refit_cost == "expensive"
        assert model.requires.supports_quantiles is True


class TestExogenasFuturas:
    def test_nhits_y_tft_las_declaran_y_patchtst_no(self) -> None:
        assert NHITSForecaster().requires.needs_futr_exog is True
        assert TFTForecaster().requires.needs_futr_exog is True
        assert PatchTSTForecaster(use_futr_exog=False).requires.needs_futr_exog is False

    def test_patchtst_rechaza_configurarse_con_exogenas_futuras(self) -> None:
        # Aceptarlas y descartarlas en silencio dejaria creer que el modelo usa
        # la temperatura cuando no puede.
        with pytest.raises(ValueError, match="univariado"):
            PatchTSTForecaster(use_futr_exog=True)

    def test_una_trama_futura_incompleta_lanza_un_error_claro(self, panel: Panel) -> None:
        fitted = NHITSForecaster(input_size=INPUT_SIZE, mlp_units=((16, 16),) * 3, **_BUDGET).fit(
            panel, h=H
        )
        incompleta = FutrFrame(
            df=_futr_frame(panel, H).df.drop(columns=["temp_c"]),
            window=None,  # type: ignore[arg-type]
            vintage=Vintage.REALIZED,
        )
        with pytest.raises(ValueError, match="faltan las exogenas futuras"):
            fitted.predict(incompleta)


@pytest.fixture(scope="module")
def fitted_tft(panel: Panel):
    """TFT ajustado y con una prediccion hecha, que es lo que fija los pesos.

    `fit` deja pesos poblados por su pasada de validacion; `predict` los
    sobreescribe con los del tramo que de verdad se quiere explicar. Es la
    precondicion que documentan `variable_importance` y `temporal_attention`.
    """
    model = TFTForecaster(input_size=INPUT_SIZE, hidden_size=8, n_head=2, **_BUDGET)
    fitted = model.fit(panel, h=H)
    fitted.predict(_futr_frame(panel, H))
    return fitted


class TestInterpretabilidadDelTFT:
    def test_los_pesos_de_seleccion_de_variables_estan_en_formato_largo(self, fitted_tft) -> None:
        importance = fitted_tft.variable_importance()

        assert list(importance.columns) == ["kind", "feature", "block", "value"]
        assert set(importance["kind"]) == {"attention_variable"}
        assert set(importance["block"]) <= {"past", "future"}

    def test_los_pesos_suman_uno_dentro_de_cada_bloque(self, fitted_tft) -> None:
        # Son la salida de un softmax sobre las variables del bloque: si no
        # sumasen uno, se estaria promediando algo que no es una seleccion.
        importance = fitted_tft.variable_importance()
        for _, block in importance.groupby("block"):
            assert block["value"].sum() == pytest.approx(1.0, abs=1e-4)

    def test_la_temperatura_aparece_en_los_dos_bloques(self, fitted_tft) -> None:
        importance = fitted_tft.variable_importance()
        past = set(importance.loc[importance["block"] == "past", "feature"])
        future = set(importance.loc[importance["block"] == "future", "feature"])
        assert "temp_c" in past
        assert "observed_target" in past
        assert future == {"temp_c"}

    def test_la_atencion_temporal_cubre_contexto_y_horizonte(self, fitted_tft) -> None:
        attention = fitted_tft.temporal_attention()

        assert list(attention.columns) == ["kind", "ds", "offset", "value"]
        assert len(attention) == INPUT_SIZE + H
        # `offset` es relativo al cutoff: negativo en el contexto, positivo en
        # el horizonte, y el paso 0 no existe porque el cutoff es el ultimo
        # instante observado (offset -0 seria el propio cutoff, offset 1 el
        # primero predicho).
        assert attention["offset"].min() == -(INPUT_SIZE - 1)
        assert attention["offset"].max() == H

    def test_la_atencion_se_ancla_al_cutoff_del_ajuste(self, fitted_tft, panel: Panel) -> None:
        attention = fitted_tft.temporal_attention()
        en_el_cutoff = attention.loc[attention["offset"] == 0, "ds"]
        assert len(en_el_cutoff) == 1
        assert pd.Timestamp(en_el_cutoff.iloc[0]) == panel.last_ds

    def test_la_atencion_es_una_distribucion(self, fitted_tft) -> None:
        attention = fitted_tft.temporal_attention()
        assert (attention["value"] >= 0).all()
        assert attention["value"].sum() == pytest.approx(1.0, abs=1e-4)

    def test_predict_sobreescribe_los_pesos_de_la_pasada_de_validacion(self, panel: Panel) -> None:
        # `fit` termina con una pasada de validacion, asi que los pesos ya
        # existen al ajustar; describen ese lote, no el tramo predicho.
        # `predict` los sobreescribe: por eso las dos funciones documentan que
        # hay que llamarlo antes si se quiere explicar la ventana de interes.
        fitted = TFTForecaster(input_size=INPUT_SIZE, hidden_size=8, n_head=2, **_BUDGET).fit(
            panel, h=H
        )
        tras_ajustar = fitted.temporal_attention()["value"].to_numpy()

        fitted.predict(_futr_frame(panel, H))
        tras_predecir = fitted.temporal_attention()["value"].to_numpy()

        assert not np.allclose(tras_ajustar, tras_predecir)

    def test_sin_ninguna_pasada_hacia_delante_el_error_lo_dice(self, panel: Panel) -> None:
        fitted = TFTForecaster(input_size=INPUT_SIZE, hidden_size=8, n_head=2, **_BUDGET).fit(
            panel, h=H
        )
        fitted.net.interpretability_params = {}
        with pytest.raises(ValueError, match="ninguna pasada hacia delante"):
            fitted.variable_importance()

    @pytest.mark.parametrize(
        "build",
        [
            lambda: NHITSForecaster(input_size=INPUT_SIZE, mlp_units=((16, 16),) * 3, **_BUDGET),
            lambda: PatchTSTForecaster(
                input_size=INPUT_SIZE,
                hidden_size=16,
                linear_hidden_size=16,
                encoder_layers=1,
                n_heads=2,
                patch_len=8,
                stride=4,
                use_futr_exog=False,
                **_BUDGET,
            ),
        ],
        ids=["nhits", "patchtst"],
    )
    def test_los_otros_dos_no_fingen_tener_interpretabilidad(self, build, panel: Panel) -> None:
        # Devolver una tabla vacia haria pasar por "sin senal" lo que en
        # realidad es "sin mecanismo".
        fitted = build().fit(panel, h=H)
        with pytest.raises(NotImplementedError):
            fitted.variable_importance()
        with pytest.raises(NotImplementedError):
            fitted.temporal_attention()


class TestValidacionDeParametros:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_size": 0},
            {"max_steps": 0},
            {"val_check_steps": 0},
            {"quantiles": (0.5, 0.1)},
            {"quantiles": (0.1, 0.9)},  # sin mediana: no hay pronostico puntual
        ],
    )
    def test_rechaza_configuraciones_invalidas(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            NHITSForecaster(**kwargs)  # type: ignore[arg-type]


class TestImportPerezoso:
    def test_sin_neuralforecast_instalado_el_mensaje_dice_que_instalar(
        self, monkeypatch: pytest.MonkeyPatch, panel: Panel
    ) -> None:
        for name in list(sys.modules):
            if name == "neuralforecast" or name.startswith("neuralforecast."):
                monkeypatch.setitem(sys.modules, name, None)

        with pytest.raises(ImportError, match="extra 'deep'"):
            NHITSForecaster(input_size=INPUT_SIZE, **_BUDGET).fit(panel, h=H)


class TestIntegracionConElMotor:
    def test_un_backtest_con_refit_por_ventana_no_produce_fuga(self, panel: Panel) -> None:
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=1)
        model = NHITSForecaster(input_size=INPUT_SIZE, mlp_units=((16, 16),) * 3, **_BUDGET)

        result = backtest(panel, [model], plan, futr=provider)

        assert (result.model_runs["status"] == "ok").all()
        assert (result.forecasts["ds"] > result.forecasts["cutoff"]).all()
        assert (result.model_runs["n_params"] > 0).all()

    def test_reutilizar_el_ajuste_falla_ruidosamente(self, panel: Panel) -> None:
        # La red emite exactamente los pasos 1..h desde su cutoff; con la
        # politica por defecto (`expensive`), la segunda ventana pide 7..12 y
        # tiene que fallar en vez de publicar un desfase.
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        plan = BacktestPlan(h=H, n_windows=2, step_size=H)
        model = NHITSForecaster(input_size=INPUT_SIZE, mlp_units=((16, 16),) * 3, **_BUDGET)

        result = backtest(panel, [model], plan, futr=provider)

        assert result.model_runs["status"].tolist() == ["ok", "failed"]
        assert "refit_every=1" in result.model_runs["error"].dropna().iloc[0]
