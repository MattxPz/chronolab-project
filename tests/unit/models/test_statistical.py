"""AutoARIMA, AutoETS, AutoTheta y MSTL: conformidad de protocolo y barreras documentadas.

Los cuatro se ejercitan sobre un panel de juguete (estacionalidad de periodo 4,
no 24/168) y con topes de orden minimos, precisamente para que la suite quede
rapida: la calidad estadistica del ajuste no es lo que se comprueba aqui —para
eso esta el backtest completo del hito de este modulo—, sino que el envoltorio
cumple el protocolo, traduce columnas correctamente y respeta las dos barreras
que describe el docstring de `chronolab.models.adapters.statsforecast`: el
`h` fijado en el constructor y el limite de horizonte de los intervalos
conformales.

Aun con esos topes minimos, ajustar de verdad sigue costando CPU real (varios
fits de `AutoARIMA` repetidos a lo largo del fichero): todo el modulo se marca
`slow` y queda fuera de `make test-fast`, igual que el resto de la suite que
paga un coste de computo real en vez de E/S.

`statsforecast` vive en el extra `ml` (D20), no en el nucleo: el job `quality`
de CI hace un `uv sync` a secas, sin ese extra, exactamente igual que el
entorno por defecto de `make test`. `pytest.importorskip` a nivel de modulo
hace que todo el fichero se salte con limpieza ahi, en vez de fallar con un
`ImportError` de cada test —el mismo patron que ya usa el cruce contra
statsforecast en `test_baselines.py`, aplicado aqui a todo el fichero porque
aqui *todo* el fichero depende de la libreria, no un solo test.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsforecast")

from chronolab.data.futr import RealizedFutrProvider
from chronolab.errors import PerfectForesightWarning
from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.models.adapters.statsforecast import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    AutoThetaForecaster,
    MSTLForecaster,
)
from chronolab.models.protocols import FittedForecaster, Forecaster
from chronolab.panel import FutrFrame, Panel, PanelSpec
from chronolab.types import DatasetId, Vintage

pytestmark = pytest.mark.slow

H = 4
SEASON = 4


def _panel(n_series: int = 2, n_hours: int = 120) -> Panel:
    rng = np.random.default_rng(1)
    index = pd.date_range("2023-01-02", periods=n_hours, freq="h")
    base = 50 + 5 * np.sin(2 * np.pi * np.arange(n_hours) / SEASON) + rng.normal(0, 0.5, n_hours)
    frame = pd.concat(
        [
            pd.DataFrame({"unique_id": f"s{i}", "ds": index, "y": base + i * 3})
            for i in range(n_series)
        ],
        ignore_index=True,
    )
    spec = PanelSpec(dataset_id=DatasetId("mini"), freq="h", seasonalities=(SEASON, SEASON * 2))
    return Panel(df=frame, spec=spec)


def _models() -> list[Forecaster]:
    return [
        AutoARIMAForecaster(h=H, season_length=SEASON, max_p=1, max_q=1, max_P=0, max_Q=0),
        AutoETSForecaster(h=H, season_length=SEASON),
        AutoThetaForecaster(h=H, season_length=SEASON),
        MSTLForecaster(h=H, season_lengths=(SEASON, SEASON * 2), trend_max_p=1, trend_max_q=1),
    ]


def _model_ids() -> list[str]:
    return ["auto_arima", "auto_ets", "auto_theta", "mstl"]


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


class TestHorizonteDeConstruccion:
    """`ConformalIntervals` fija `h` al construir el modelo, no al llamar a `fit`."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda: AutoARIMAForecaster(h=H, season_length=SEASON),
            lambda: AutoETSForecaster(h=H, season_length=SEASON),
            lambda: AutoThetaForecaster(h=H, season_length=SEASON),
            lambda: MSTLForecaster(h=H, season_lengths=(SEASON, SEASON * 2)),
        ],
        ids=_model_ids(),
    )
    def test_fit_rechaza_un_h_distinto_del_construido(self, build, panel: Panel) -> None:
        model = build()
        with pytest.raises(ValueError, match="se construyo con h="):
            model.fit(panel, h=H + 1)


class TestValidacionDeParametros:
    @pytest.mark.parametrize(
        "build",
        [
            lambda: AutoARIMAForecaster(h=H, levels=(80, 101)),
            lambda: AutoETSForecaster(h=H, levels=(0, 95)),
            lambda: AutoThetaForecaster(h=H, levels=(80, 100)),
            lambda: MSTLForecaster(h=H, levels=(-5, 95)),
        ],
        ids=_model_ids(),
    )
    def test_rechaza_niveles_fuera_de_rango(self, build) -> None:
        with pytest.raises(ValueError, match="nivel de intervalo fuera de"):
            build()

    @pytest.mark.parametrize(
        "build",
        [
            lambda: AutoARIMAForecaster(h=H, calibration_windows=1),
            lambda: AutoETSForecaster(h=H, calibration_windows=0),
            lambda: AutoThetaForecaster(h=H, calibration_windows=1),
            lambda: MSTLForecaster(h=H, calibration_windows=1),
        ],
        ids=_model_ids(),
    )
    def test_rechaza_menos_de_dos_ventanas_de_calibracion(self, build) -> None:
        # ConformalIntervals de statsforecast exige explicitamente al menos dos.
        with pytest.raises(ValueError, match="calibration_windows debe ser >= 2"):
            build()

    def test_mstl_rechaza_periodos_vacios_o_invalidos(self) -> None:
        with pytest.raises(ValueError, match="no puede estar vacia"):
            MSTLForecaster(h=H, season_lengths=())
        with pytest.raises(ValueError, match="cada periodo debe ser >= 2"):
            MSTLForecaster(h=H, season_lengths=(1, 24))


class TestMinContext:
    def test_con_intervalos_el_minimo_lo_domina_la_calibracion_conformal(self) -> None:
        # calibration_windows * h + 1 = 2*4+1 = 9, frente a 2*season_length = 8:
        # gana la calibracion conformal.
        model = AutoETSForecaster(h=H, season_length=SEASON, calibration_windows=2)
        assert model.requires.min_context == 9

    def test_sin_intervalos_el_minimo_es_solo_el_de_las_estaciones(self) -> None:
        model = AutoETSForecaster(h=H, season_length=SEASON, use_intervals=False)
        assert model.requires.min_context == 2 * SEASON

    def test_mstl_usa_el_periodo_mas_largo(self) -> None:
        model = MSTLForecaster(h=H, season_lengths=(4, 8), use_intervals=False)
        assert model.requires.min_context == 16


class TestUsoIntervalsDesactivado:
    def test_sin_intervalos_solo_queda_la_mediana_como_pronostico_puntual(
        self, panel: Panel
    ) -> None:
        # q_5000 no exige haber calibrado ningun intervalo: es el pronostico
        # puntual con otro nombre. Los extremos si lo exigen, y por eso no
        # aparecen sin `use_intervals`.
        model = AutoETSForecaster(h=H, season_length=SEASON, use_intervals=False)
        fitted = model.fit(panel, h=H)
        prediction = fitted.predict()

        assert list(prediction.columns) == ["unique_id", "ds", "y_hat", "q_5000"]
        pd.testing.assert_series_equal(prediction["q_5000"], prediction["y_hat"], check_names=False)

    def test_sin_intervalos_el_ajuste_se_puede_reutilizar_mas_alla_de_h(self, panel: Panel) -> None:
        model = AutoETSForecaster(h=H, season_length=SEASON, use_intervals=False)
        fitted = model.fit(panel, h=H)

        # H + 2 pasos mas alla del ajuste original: sin intervalos, no hay limite.
        far = FutrFrame(
            df=pd.DataFrame(
                {
                    "unique_id": [uid for uid in panel.ids() for _ in range(H + 2)],
                    "ds": [
                        panel.last_ds + pd.Timedelta(hours=k)
                        for _ in panel.ids()
                        for k in range(1, H + 3)
                    ],
                }
            ),
            window=None,  # type: ignore[arg-type]
            vintage=Vintage.REALIZED,
        )
        prediction = fitted.predict(far)
        assert len(prediction) == (H + 2) * len(panel.ids())


class TestLimiteDeHorizonteConformal:
    """Reutilizar un ajuste con intervalos mas alla de `h` esta documentado, no silenciado."""

    def test_pedir_mas_de_h_pasos_con_intervalos_lanza_un_error_explicativo(
        self, panel: Panel
    ) -> None:
        model = AutoETSForecaster(h=H, season_length=SEASON)
        fitted = model.fit(panel, h=H)

        far = FutrFrame(
            df=pd.DataFrame(
                {
                    "unique_id": [uid for uid in panel.ids() for _ in range(H + 1)],
                    "ds": [
                        panel.last_ds + pd.Timedelta(hours=k)
                        for _ in panel.ids()
                        for k in range(1, H + 2)
                    ],
                }
            ),
            window=None,  # type: ignore[arg-type]
            vintage=Vintage.REALIZED,
        )
        with pytest.raises(ValueError, match="refit_every=1"):
            fitted.predict(far)


class TestImportPerezoso:
    """El modulo se importa sin el extra `ml`; solo `fit` lo necesita de verdad."""

    def test_sin_statsforecast_instalado_el_mensaje_dice_que_instalar(
        self, monkeypatch: pytest.MonkeyPatch, panel: Panel
    ) -> None:
        for name in list(sys.modules):
            if name == "statsforecast" or name.startswith("statsforecast."):
                monkeypatch.setitem(sys.modules, name, None)

        model = AutoETSForecaster(h=H, season_length=SEASON)
        with pytest.raises(ImportError, match="extra 'ml'"):
            model.fit(panel, h=H)


class TestIntegracionConElMotor:
    def test_un_backtest_con_refit_every_uno_no_produce_fuga(self, panel: Panel) -> None:
        # refit_every=1: cada ventana reajusta con su propio h, que es lo que
        # exige tener intervalos conformales validos en todas las ventanas.
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=1)
        model = AutoETSForecaster(h=H, season_length=SEASON)

        result = backtest(panel, [model], plan)

        assert (result.model_runs["status"] == "ok").all()
        assert (result.forecasts["ds"] > result.forecasts["cutoff"]).all()
        assert not result.forecasts[["q_0250", "q_9750"]].isna().any().any()

    def test_reutilizar_sin_refit_every_explicito_falla_a_partir_de_la_segunda_ventana(
        self, panel: Panel
    ) -> None:
        # refit_cost="expensive" hace que la politica por defecto sea un solo
        # ajuste para todo el run; con intervalos activos eso choca con el
        # limite de horizonte documentado, y el motor lo registra como fallo
        # de modelo (A6), no como fuga. Hace falta un FutrProvider para que el
        # ajuste reutilizado sepa a que instantes le toca predecir en la
        # segunda ventana: sin el, fallaria antes por el motivo generico de
        # cualquier modelo sin exogenas (backtest.py), no por el limite de
        # horizonte especifico de los intervalos conformales que este test
        # quiere comprobar.
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        plan = BacktestPlan(h=H, n_windows=2, step_size=H)
        model = AutoETSForecaster(h=H, season_length=SEASON)

        result = backtest(panel, [model], plan, futr=provider)

        assert result.model_runs["status"].tolist() == ["ok", "failed"]
        assert "refit_every=1" in result.model_runs["error"].dropna().iloc[0]


class TestConProveedorDeExogenasFuturas:
    """Aunque no las usan, aceptan un `FutrProvider` en el run sin romperse."""

    def test_backtest_con_futrprovider_sigue_funcionando(self, panel: Panel) -> None:
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=1)
        result = backtest(
            panel, [AutoARIMAForecaster(h=H, season_length=SEASON)], plan, futr=provider
        )
        assert (result.model_runs["status"] == "ok").all()


class TestHuecosEnLaObjetivo:
    """`handles_nan_target=False` obliga al adaptador a imputar dentro de `fit`.

    El invariante I3 conserva un hueco como ``y = NaN`` en una fila que si
    existe, y `mstl` aborta con "cannot handle missing values" en cuanto ve uno.
    Sin la imputacion del adaptador, un unico hueco en el tramo de
    entrenamiento no degrada el ajuste: elimina la ventana entera del run.
    """

    @staticmethod
    def _panel_con_hueco(*, leading: bool = False) -> Panel:
        panel = _panel()
        frame = panel.df.copy()
        target = panel.spec.target
        rows = frame.index[frame["unique_id"] == "s0"]
        frame.loc[rows[:3] if leading else rows[20:26], target] = np.nan
        return Panel(df=frame, spec=panel.spec)

    @pytest.mark.parametrize("model", _models(), ids=_model_ids())
    def test_un_hueco_interior_no_tumba_el_ajuste(self, model: Forecaster) -> None:
        fitted = model.fit(self._panel_con_hueco(), h=H)
        prediction = fitted.predict()
        assert len(prediction) == 2 * H
        assert prediction["y_hat"].notna().all()

    def test_un_hueco_al_principio_tampoco(self) -> None:
        # El relleno hacia delante no tiene pasado del que tirar en la primera
        # fila; se cubre con la media del propio tramo de entrenamiento.
        model = MSTLForecaster(
            h=H, season_lengths=(SEASON, SEASON * 2), trend_max_p=1, trend_max_q=1
        )
        prediction = model.fit(self._panel_con_hueco(leading=True), h=H).predict()
        assert prediction["y_hat"].notna().all()

    def test_el_relleno_solo_mira_hacia_atras(self) -> None:
        # La imputacion admisible del proyecto es la retrospectiva: el valor de
        # un hueco tiene que ser el ultimo observado antes de el, nunca el
        # siguiente, que en el instante del hueco todavia no existia.
        from chronolab.models.adapters.statsforecast import _univariate_frame

        frame = _univariate_frame(self._panel_con_hueco())
        own = frame.loc[frame["unique_id"] == "s0", "y"].to_numpy()
        assert np.isfinite(own).all()
        assert (own[20:26] == own[19]).all()

    def test_el_backtest_no_registra_ventanas_fallidas(self) -> None:
        model = MSTLForecaster(
            h=H, season_lengths=(SEASON, SEASON * 2), trend_max_p=1, trend_max_q=1
        )
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=1)
        result = backtest(self._panel_con_hueco(), [model], plan)
        assert (result.model_runs["status"] == "ok").all()
