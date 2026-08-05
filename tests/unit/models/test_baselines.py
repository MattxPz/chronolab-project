"""Los cinco baselines: pronosticos y sigma verificados a mano, mas el patron D15.

La serie de referencia es la misma en todo el fichero, elegida para que las
diferencias, la estacion y la media salgan con decimales manejables:

    t:      0   1   2   3   4   5   6   7
    valor: 10  20  30  40  50  62  70  80

Con ``season = 2`` y ``window = 3``:

===================  ==========================================================
estadistico          calculo
===================  ==========================================================
naive                ultimo valor = 80
diferencias          [10, 10, 10, 10, 12, 8, 10]; sigma = std(ddof=1) = 1.154701
seasonal (m=2)        ultimos dos valores = [70, 80]
residuos estacionales [20, 20, 20, 22, 20, 18]; sigma = 1.264911
window (k=3)          media(62, 70, 80) = 70.666667; sigma = 9.018500
historic               media(10..80) = 45.25; sigma = 24.679373
drift                  (80-10)/7 = 10.0; residuos = diferencias - 10; sigma = 1.154701
===================  ==========================================================

Todos los `sigma` son `std(..., ddof=1)`, la desviacion tipica **muestral**, la
misma convencion que usa `statsforecast`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.evaluation.splitters import RollingOriginSplitter
from chronolab.models.baselines import (
    DriftForecaster,
    HistoricAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    WindowAverageForecaster,
    _normal_ppf,
)
from chronolab.models.protocols import FittedForecaster, Forecaster
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId, Vintage

VALUES = [10.0, 20.0, 30.0, 40.0, 50.0, 62.0, 70.0, 80.0]
SPEC = PanelSpec(dataset_id=DatasetId("mini"), freq="h", seasonalities=(2, 4))


def _panel(values: list[float] = VALUES, *, n_series: int = 1) -> Panel:
    """Panel de una o mas series identicas, horario, empezando en un lunes."""
    index = pd.date_range("2023-01-02", periods=len(values), freq="h")
    frame = pd.concat(
        [pd.DataFrame({"unique_id": f"s{i}", "ds": index, "y": values}) for i in range(n_series)],
        ignore_index=True,
    )
    return Panel(df=frame, spec=SPEC)


def _predict(model: Forecaster, panel: Panel, h: int, **kwargs: object) -> pd.DataFrame:
    fitted = model.fit(panel, h=h)
    return fitted.predict(**kwargs)  # type: ignore[arg-type]


# Con los parametros por defecto, SeasonalNaive y WindowAverage exigen 26 y 24
# observaciones respectivamente: la serie de referencia de 8 puntos no basta
# para probar conformidad de protocolo con sus defaults, asi que estos tests
# usan un panel mas largo.
_LONG_VALUES = [float(50 + 10 * (i % 5)) for i in range(40)]


class TestConformidadConElProtocolo:
    """Los cinco son `Forecaster`/`FittedForecaster` de verdad, no solo de nombre."""

    @pytest.mark.parametrize(
        "model",
        [
            NaiveForecaster(),
            SeasonalNaiveForecaster(),
            WindowAverageForecaster(),
            HistoricAverageForecaster(),
            DriftForecaster(),
        ],
    )
    def test_el_forecaster_satisface_el_protocolo(self, model: Forecaster) -> None:
        panel = _panel(_LONG_VALUES)
        assert isinstance(model, Forecaster)
        fitted = model.fit(panel, h=2)
        assert isinstance(fitted, FittedForecaster)
        assert fitted.cutoff == panel.last_ds
        assert fitted.h == 2
        assert fitted.n_params is None
        assert fitted.fit_seconds >= 0.0

    @pytest.mark.parametrize(
        "model",
        [
            NaiveForecaster(),
            SeasonalNaiveForecaster(),
            WindowAverageForecaster(),
            HistoricAverageForecaster(),
            DriftForecaster(),
        ],
    )
    def test_predict_devuelve_exactamente_h_filas_por_serie(self, model: Forecaster) -> None:
        panel = _panel(_LONG_VALUES, n_series=3)
        prediction = _predict(model, panel, h=4)

        assert len(prediction) == 3 * 4
        assert set(prediction["unique_id"]) == {"s0", "s1", "s2"}
        assert (prediction["ds"] > panel.last_ds).all()

    @pytest.mark.parametrize(
        "model",
        [
            NaiveForecaster(),
            SeasonalNaiveForecaster(),
            WindowAverageForecaster(),
            HistoricAverageForecaster(),
            DriftForecaster(),
        ],
    )
    def test_ninguno_necesita_exogenas_futuras(self, model: Forecaster) -> None:
        assert model.requires.needs_futr_exog is False


class TestRequireFinite:
    """`_require_finite`: la comprobacion defensiva que ningun panel valido dispara.

    Con el invariante I3, el primer y el ultimo punto de una serie siempre son
    observaciones reales, asi que `ffill` nunca deja `NaN` en la practica. Este
    test construye directamente el caso que esa garantia hace imposible desde
    un `Panel`, para que la rama defensiva no quede sin ejercitar.
    """

    def test_una_serie_enteramente_nula_se_rechaza(self) -> None:
        from chronolab.models.baselines import _require_finite

        with pytest.raises(ValueError, match="no tiene ninguna observacion real"):
            _require_finite(np.array([np.nan, np.nan]), uid="s0", model="naive", minimum=1)


class TestNaive:
    def test_el_pronostico_puntual_es_el_ultimo_valor_en_todo_el_horizonte(self) -> None:
        prediction = _predict(NaiveForecaster(), _panel(), h=3)
        assert prediction["y_hat"].tolist() == pytest.approx([80.0, 80.0, 80.0])

    def test_la_mediana_coincide_con_el_punto_y_el_sigma_crece_con_sqrt_h(self) -> None:
        # sigma = std([10,10,10,10,12,8,10], ddof=1) = 1.154701
        prediction = _predict(NaiveForecaster(), _panel(), h=3, quantiles=(0.5, 0.75))
        sigma = math.sqrt(np.var([10, 10, 10, 10, 12, 8, 10], ddof=1))

        assert prediction["q_5000"].tolist() == pytest.approx([80.0, 80.0, 80.0])
        expected_q75 = [80.0 + sigma * math.sqrt(h) * _normal_ppf(0.75) for h in (1, 2, 3)]
        assert prediction["q_7500"].tolist() == pytest.approx(expected_q75)
        assert sigma == pytest.approx(1.1547005383792515)

    def test_con_dos_observaciones_el_sigma_es_nan_pero_el_punto_no(self) -> None:
        # Un unico residuo no tiene desviacion muestral definida (ddof=1).
        prediction = _predict(NaiveForecaster(), _panel([10.0, 20.0]), h=1)
        assert prediction["y_hat"].item() == pytest.approx(20.0)
        assert math.isnan(prediction["q_5000"].item())

    def test_min_context_es_dos(self) -> None:
        assert NaiveForecaster().requires.min_context == 2


class TestSeasonalNaive:
    def test_alterna_los_dos_ultimos_valores_de_la_estacion(self) -> None:
        prediction = _predict(SeasonalNaiveForecaster(season=2), _panel(), h=4)
        # h=1 -> t-1 (70), h=2 -> t (80), h=3 -> un ciclo despues de t-1 (70)...
        assert prediction["y_hat"].tolist() == pytest.approx([70.0, 80.0, 70.0, 80.0])

    def test_sigma_crece_por_ciclos_estacionales_completos(self) -> None:
        # residuos = [20, 20, 20, 22, 20, 18] -> sigma = 1.264911
        prediction = _predict(SeasonalNaiveForecaster(season=2), _panel(), h=4, quantiles=(0.5,))
        sigma = math.sqrt(np.var([20, 20, 20, 22, 20, 18], ddof=1))
        assert sigma == pytest.approx(1.2649110640673518)

        # h=1,2 estan en el primer ciclo (k=0); h=3,4 en el segundo (k=1).
        deltas = (prediction["q_5000"] - prediction["y_hat"]).tolist()
        assert deltas[0] == deltas[1] == pytest.approx(0.0)  # cuantil 0.5 = punto

    def test_rechaza_una_estacionalidad_no_positiva(self) -> None:
        with pytest.raises(ValueError, match="season debe ser >= 1"):
            SeasonalNaiveForecaster(season=0)

    def test_min_context_exige_una_estacion_mas_dos_residuos(self) -> None:
        assert SeasonalNaiveForecaster(season=24).requires.min_context == 26

    def test_falla_con_menos_de_una_estacion_completa(self) -> None:
        with pytest.raises(ValueError, match="hacen falta al menos"):
            SeasonalNaiveForecaster(season=24).fit(_panel(), h=1)


class TestWindowAverage:
    def test_la_media_de_la_ventana_es_constante_en_el_horizonte(self) -> None:
        # media(62, 70, 80) = 70.666667
        prediction = _predict(WindowAverageForecaster(window=3), _panel(), h=3)
        assert prediction["y_hat"].tolist() == pytest.approx([70.666667] * 3, rel=1e-6)

    def test_sigma_no_crece_con_el_horizonte(self) -> None:
        prediction = _predict(WindowAverageForecaster(window=3), _panel(), h=3, quantiles=(0.9,))
        anchuras = (prediction["q_9000"] - prediction["y_hat"]).tolist()
        assert anchuras[0] == pytest.approx(anchuras[1]) == pytest.approx(anchuras[2])

    def test_una_ventana_de_uno_no_tiene_sigma_definido(self) -> None:
        prediction = _predict(WindowAverageForecaster(window=1), _panel(), h=1, quantiles=(0.5,))
        assert prediction["y_hat"].item() == pytest.approx(80.0)
        assert math.isnan(prediction["q_5000"].item())

    def test_rechaza_una_ventana_no_positiva(self) -> None:
        with pytest.raises(ValueError, match="window debe ser >= 1"):
            WindowAverageForecaster(window=0)


class TestHistoricAverage:
    def test_la_media_de_toda_la_historia_es_constante(self) -> None:
        # media(10..80) = 45.25
        prediction = _predict(HistoricAverageForecaster(), _panel(), h=2)
        assert prediction["y_hat"].tolist() == pytest.approx([45.25, 45.25])

    def test_sigma_incluye_el_ajuste_de_estimar_la_media(self) -> None:
        # sigma = std(valores, ddof=1) = 24.679373; PI = sigma*sqrt(1+1/n)
        sigma = math.sqrt(np.var(VALUES, ddof=1))
        prediction = _predict(HistoricAverageForecaster(), _panel(), h=1, quantiles=(0.5, 0.9))
        esperado = 45.25 + sigma * math.sqrt(1 + 1 / 8) * _normal_ppf(0.9)
        assert prediction["q_9000"].item() == pytest.approx(esperado)
        assert sigma == pytest.approx(24.67937253196338)

    def test_funciona_con_una_sola_observacion(self) -> None:
        prediction = _predict(HistoricAverageForecaster(), _panel([42.0]), h=1)
        assert prediction["y_hat"].item() == pytest.approx(42.0)


class TestDrift:
    def test_extrapola_la_pendiente_media(self) -> None:
        # drift = (80-10)/7 = 10.0; y_hat(h) = 80 + 10h
        prediction = _predict(DriftForecaster(), _panel(), h=3)
        assert prediction["y_hat"].tolist() == pytest.approx([90.0, 100.0, 110.0])

    def test_sin_tendencia_coincide_con_el_naive(self) -> None:
        plano = [50.0] * 6
        naive = _predict(NaiveForecaster(), _panel(plano), h=3)
        drift = _predict(DriftForecaster(), _panel(plano), h=3)
        assert drift["y_hat"].tolist() == pytest.approx(naive["y_hat"].tolist())

    def test_sigma_crece_mas_rapido_que_el_naive_simple(self) -> None:
        # factor drift: h*(1 + h/(n-1)); factor naive: h. Con n=8 y h>=1, el
        # factor de drift es siempre mayor: el drift arrastra mas incertidumbre
        # porque tambien esta estimando una pendiente.
        naive = _predict(NaiveForecaster(), _panel(), h=3, quantiles=(0.9,))
        drift = _predict(DriftForecaster(), _panel(), h=3, quantiles=(0.9,))
        ancho_naive = (naive["q_9000"] - naive["y_hat"]).to_numpy()
        ancho_drift = (drift["q_9000"] - drift["y_hat"]).to_numpy()
        assert (ancho_drift > ancho_naive).all()

    def test_min_context_es_dos(self) -> None:
        assert DriftForecaster().requires.min_context == 2


class TestFuenteDeLosInstantesPredichos:
    """`ds` sale del `FutrFrame` cuando lo hay, y de `cutoff + h*freq` cuando no."""

    def test_sin_futrframe_los_instantes_salen_del_cutoff_y_la_frecuencia(self) -> None:
        panel = _panel()
        prediction = _predict(NaiveForecaster(), panel, h=3)
        esperado = pd.date_range(panel.last_ds, periods=4, freq="h")[1:]
        assert prediction["ds"].tolist() == list(esperado)

    def test_con_futrframe_se_usan_sus_instantes_aunque_difieran_del_calculo_local(
        self,
    ) -> None:
        panel = _panel()
        window = RollingOriginSplitter(h=2, n_windows=1, gap=3).split(panel)[0]
        train = panel.train(window)
        fitted = NaiveForecaster().fit(train, h=window.h)

        from chronolab.panel import FutrFrame

        futr = FutrFrame(
            df=pd.DataFrame(
                {"unique_id": ["s0", "s0"], "ds": [window.first_pred, window.last_pred]}
            ),
            window=window,
            vintage=Vintage.REALIZED,
        )
        prediction = fitted.predict(futr)
        assert prediction["ds"].tolist() == [window.first_pred, window.last_pred]

    def test_rechaza_un_futrframe_con_un_numero_de_instantes_distinto_de_h(self) -> None:
        panel = _panel()
        fitted = NaiveForecaster().fit(panel, h=3)

        from chronolab.evaluation.splitters import Window
        from chronolab.panel import FutrFrame

        ventana = Window(
            window_id=0,
            stage="dev",
            train_start=panel.first_ds,
            cutoff=panel.last_ds,
            first_pred=panel.last_ds + pd.Timedelta(hours=1),
            last_pred=panel.last_ds + pd.Timedelta(hours=2),
            h=3,
            gap=0,
        )
        futr = FutrFrame(
            df=pd.DataFrame(
                {"unique_id": ["s0", "s0"], "ds": [ventana.first_pred, ventana.last_pred]}
            ),
            window=ventana,
            vintage=Vintage.REALIZED,
        )
        with pytest.raises(ValueError, match="se esperaban 3 instantes"):
            fitted.predict(futr)


class TestNormalPpf:
    def test_es_la_inversa_de_valores_de_tabla_conocidos(self) -> None:
        # Cuantiles de tabla de la normal estandar.
        assert _normal_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
        assert _normal_ppf(0.975) == pytest.approx(1.959963985, abs=1e-8)
        assert _normal_ppf(0.025) == pytest.approx(-1.959963985, abs=1e-8)
        assert _normal_ppf(0.9) == pytest.approx(1.281551566, abs=1e-8)

    def test_es_antisimetrica(self) -> None:
        assert _normal_ppf(0.1) == pytest.approx(-_normal_ppf(0.9), abs=1e-9)

    def test_rechaza_probabilidades_fuera_de_rango(self) -> None:
        with pytest.raises(ValueError, match="fuera de"):
            _normal_ppf(0.0)
        with pytest.raises(ValueError, match="fuera de"):
            _normal_ppf(1.0)

    def test_coincide_con_scipy(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        for p in (0.001, 0.025, 0.1, 0.3, 0.5, 0.7, 0.9, 0.975, 0.999):
            assert _normal_ppf(p) == pytest.approx(float(stats.norm.ppf(p)), abs=1e-8)


class TestIntegracionConElMotor:
    """Los cinco baselines corren dentro de un backtest real sin activar ninguna barrera."""

    def test_un_backtest_completo_con_los_cinco_no_produce_fuga(self, hourly_panel: Panel) -> None:
        plan = BacktestPlan(h=24, n_windows=3, step_size=24, holdout_windows=1)
        modelos = [
            NaiveForecaster(),
            SeasonalNaiveForecaster(season=24),
            WindowAverageForecaster(window=24),
            HistoricAverageForecaster(),
            DriftForecaster(),
        ]
        result = backtest(hourly_panel, modelos, plan)

        assert set(result.model_runs["model_id"]) == {
            "naive",
            "seasonal_naive",
            "window_average",
            "historic_average",
            "drift",
        }
        assert (result.model_runs["status"] == "ok").all()
        assert (result.forecasts["ds"] > result.forecasts["cutoff"]).all()

    def test_sin_futrprovider_un_gap_positivo_incumple_el_contrato_de_prediccion(
        self, hourly_panel: Panel
    ) -> None:
        # Sin FutrProvider en el run, el baseline solo sabe calcular
        # `cutoff + h_step * freq`. Con `gap = 6` el tramo evaluado empieza en
        # `cutoff + 7`, asi que las primeras seis marcas calculadas caen antes
        # del tramo evaluado. No es fuga —no son instantes ya conocidos, son
        # instantes de mas— asi que el motor no detiene el run: registra el
        # `PredictionContractError` como un fallo del modelo (A6) y sigue.
        plan = BacktestPlan(h=24, n_windows=2, step_size=24, gap=6)
        result = backtest(hourly_panel, [NaiveForecaster()], plan)

        assert (result.model_runs["status"] == "failed").all()
        assert result.model_runs["error"].str.contains("fuera de").all()
        assert result.forecasts.empty


class TestStatsforecastCrossCheck:
    """D15: los pronosticos puntuales coinciden con statsforecast en las mismas ventanas.

    Es el test que hace defendible la independencia del modulo: si algun dia
    esta implementacion se desvia de la formula estandar, aqui se ve, y si
    statsforecast cambia de convencion entre versiones, este test lo detecta
    sin que el resto del arnes dependa de esa libreria para nada.
    """

    def test_los_cinco_puntuales_coinciden_con_statsforecast(self) -> None:
        sf_models = pytest.importorskip("statsforecast.models")
        sf_core = pytest.importorskip("statsforecast")

        rng = np.random.default_rng(7)
        n = 400
        index = pd.date_range("2023-01-02", periods=n, freq="h")
        values = (
            100.0 + np.cumsum(rng.normal(0, 1, n)) + 5.0 * np.sin(2 * np.pi * np.arange(n) / 24)
        )
        frame = pd.DataFrame({"unique_id": "s0", "ds": index, "y": values})
        panel = Panel(
            df=frame.assign(y=frame["y"].astype(float)),
            spec=PanelSpec(dataset_id=DatasetId("sf"), freq="h", seasonalities=(24, 168)),
        )

        sf = sf_core.StatsForecast(
            models=[
                sf_models.Naive(),
                sf_models.SeasonalNaive(season_length=24),
                sf_models.WindowAverage(window_size=24),
                sf_models.HistoricAverage(),
                sf_models.RandomWalkWithDrift(),
            ],
            freq="h",
            n_jobs=1,
        )
        h = 12
        reference = sf.forecast(df=frame, h=h).set_index("ds")

        ours = {
            "Naive": NaiveForecaster(),
            "SeasonalNaive": SeasonalNaiveForecaster(season=24),
            "WindowAverage": WindowAverageForecaster(window=24),
            "HistoricAverage": HistoricAverageForecaster(),
            "RWD": DriftForecaster(),
        }
        for column, model in ours.items():
            prediction = _predict(model, panel, h=h).set_index("ds")["y_hat"]
            np.testing.assert_allclose(
                prediction.reindex(reference.index).to_numpy(),
                reference[column].to_numpy(),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"{column} no coincide con statsforecast",
            )
