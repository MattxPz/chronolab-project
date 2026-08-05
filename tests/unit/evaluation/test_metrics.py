"""Metricas de prediccion, cada una contra un valor calculado a mano.

El ejemplo base es deliberadamente pequeno y de cabeza:

===========  ====  ====  ====  ====
observado      10    20    30    40
predicho       12    18    33    39
|error|         2     2     3     1
===========  ====  ====  ====  ====

De ahi salen MAE = 2, RMSE = sqrt(18/4), MAPE = 10.625 % y sMAPE = 10.1909 %,
y los tres primeros se pueden rehacer con una calculadora en menos de un minuto.
Comprobar contra una reimplementacion de la formula no probaria nada: si la
formula estuviese mal, el test lo estaria igual.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chronolab.errors import UnstableMetricWarning
from chronolab.evaluation.metrics import (
    crps_discrete,
    empirical_coverage,
    interval_levels,
    interval_width,
    mae,
    mape,
    mase,
    mase_denominators,
    pinball_loss,
    point_metrics,
    probabilistic_metrics,
    rmse,
    seasonal_naive_mae,
    smape,
)
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId

Y = [10.0, 20.0, 30.0, 40.0]
Y_HAT = [12.0, 18.0, 33.0, 39.0]


class TestMetricasPuntuales:
    def test_mae(self) -> None:
        # (2 + 2 + 3 + 1) / 4
        assert mae(Y, Y_HAT) == pytest.approx(2.0)

    def test_rmse(self) -> None:
        # sqrt((4 + 4 + 9 + 1) / 4) = sqrt(4.5)
        assert rmse(Y, Y_HAT) == pytest.approx(2.1213203435596424)

    def test_rmse_penaliza_mas_los_errores_grandes_que_mae(self) -> None:
        # Mismo MAE, distinto reparto: un fallo de 4 y tres aciertos frente a
        # cuatro fallos de 1. RMSE los separa y MAE no, y por eso van las dos.
        concentrado = rmse([0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0])
        repartido = rmse([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        assert mae([0.0] * 4, [4.0, 0.0, 0.0, 0.0]) == mae([0.0] * 4, [1.0] * 4) == 1.0
        assert concentrado == pytest.approx(2.0)
        assert repartido == pytest.approx(1.0)

    def test_mape(self) -> None:
        # (2/10 + 2/20 + 3/30 + 1/40) / 4 = 0.10625
        assert mape(Y, Y_HAT) == pytest.approx(10.625)

    def test_smape(self) -> None:
        # (4/22 + 4/38 + 6/63 + 2/79) / 4, en porcentaje
        assert smape(Y, Y_HAT) == pytest.approx(10.19089726618041)

    def test_smape_esta_acotada_a_200(self) -> None:
        # La cota superior es la propiedad que distingue esta definicion de las
        # otras tres que circulan con el mismo nombre.
        assert smape([1.0], [-1.0]) == pytest.approx(200.0)
        assert smape([5.0], [5.0]) == pytest.approx(0.0)

    def test_una_prediccion_perfecta_anula_las_metricas(self) -> None:
        assert mae(Y, Y) == 0.0
        assert rmse(Y, Y) == 0.0
        assert mape(Y, Y) == 0.0
        assert smape(Y, Y) == 0.0

    def test_los_pares_incompletos_se_descartan(self) -> None:
        # El hueco del panel es NaN explicito (I3): la metrica lo salta, no lo
        # imputa, y el resto del calculo es el del ejemplo base sin ese punto.
        con_hueco = mae([10.0, 20.0, float("nan"), 40.0], [12.0, 18.0, 33.0, 39.0])
        assert con_hueco == pytest.approx((2 + 2 + 1) / 3)

    def test_sin_observaciones_validas_devuelve_nan(self) -> None:
        assert math.isnan(mae([float("nan")], [1.0]))
        assert math.isnan(rmse([], []))

    def test_rechaza_longitudes_distintas(self) -> None:
        with pytest.raises(ValueError, match="misma longitud"):
            mae([1.0, 2.0], [1.0])


class TestMapeInestable:
    def test_avisa_cuando_la_serie_pasa_cerca_de_cero(self) -> None:
        # 0.01 es el 0.013 % de la magnitud media: por debajo del umbral, y su
        # error relativo domina la media entera.
        with pytest.warns(UnstableMetricWarning, match="pasa por cero"):
            valor = mape([100.0, 100.0, 0.01, 100.0], [101.0, 99.0, 0.02, 100.0])
        # (1/100 + 1/100 + 0.01/0.01 + 0) / 4 = 25.5 %
        assert valor == pytest.approx(25.5)

    def test_avisa_y_descarta_los_ceros_exactos(self) -> None:
        with pytest.warns(UnstableMetricWarning, match="observaciones nulas"):
            valor = mape([100.0, 0.0, 100.0], [110.0, 5.0, 90.0])
        # El cero no esta definido: quedan dos observaciones al 10 %.
        assert valor == pytest.approx(10.0)

    def test_todos_ceros_es_nan(self) -> None:
        with pytest.warns(UnstableMetricWarning):
            assert math.isnan(mape([0.0, 0.0], [1.0, 2.0]))

    def test_no_avisa_cuando_la_serie_esta_lejos_de_cero(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UnstableMetricWarning)
            assert mape(Y, Y_HAT) == pytest.approx(10.625)


class TestMase:
    """El denominador sale del **entrenamiento**. Es el error clasico del dominio."""

    def test_el_denominador_es_el_mae_del_naive_estacional_del_train(self) -> None:
        # train = [10, 20, 30, 40, 50, 62], m = 2
        # diferencias estacionales: |30-10|, |40-20|, |50-30|, |62-40|
        #                         =    20  ,   20   ,   20   ,   22
        # q = (20 + 20 + 20 + 22) / 4 = 20.5
        assert seasonal_naive_mae([10, 20, 30, 40, 50, 62], season=2) == pytest.approx(20.5)

    def test_mase_es_el_mae_del_test_dividido_por_ese_denominador(self) -> None:
        entrenamiento = [10, 20, 30, 40, 50, 62]
        q = seasonal_naive_mae(entrenamiento, season=2)

        # MAE del test = 2.0 (el ejemplo base), q = 20.5
        assert mase(Y, Y_HAT, denominator=q) == pytest.approx(2.0 / 20.5)
        assert mase(Y, Y_HAT, denominator=q) == pytest.approx(0.0975609756097561)

    def test_calcular_el_denominador_sobre_el_test_da_otro_numero(self) -> None:
        # Este es el bug que el test existe para atrapar. El tramo evaluado es
        # casi plano, asi que su naive estacional casi no se equivoca y el MASE
        # calculado con el saldria disparado. La diferencia entre las dos cifras
        # es de dos ordenes de magnitud: si alguien invierte el argumento, no
        # hay forma de que pase desapercibido.
        entrenamiento = [10.0, 20.0, 30.0, 40.0, 50.0, 62.0]
        observado = [70.0, 71.0, 70.0, 71.0]
        predicho = [72.0, 69.0, 72.0, 69.0]

        correcto = mase(
            observado, predicho, denominator=seasonal_naive_mae(entrenamiento, season=2)
        )
        assert correcto == pytest.approx(2.0 / 20.5)

        # El denominador "de test" seria cero —la serie se repite cada dos
        # pasos— y MASE quedaria indefinida.
        with pytest.warns(UnstableMetricWarning, match="no comete ningun error"):
            denominador_de_test = seasonal_naive_mae(observado, season=2)
        assert math.isnan(denominador_de_test)

    def test_el_denominador_por_fila_escala_antes_de_promediar(self) -> None:
        # Dos ventanas con escalas distintas: MASE es la media de los errores
        # escalados, no el error medio dividido por el denominador medio.
        errores_escalados = mase([10.0, 20.0], [11.0, 24.0], denominator=[1.0, 4.0])
        assert errores_escalados == pytest.approx((1 / 1 + 4 / 4) / 2)
        # Promediar primero daria (1 + 4) / 2 / ((1 + 4) / 2) = 1.0, otro numero.
        assert errores_escalados == pytest.approx(1.0)

        distinto = mase([10.0, 20.0], [12.0, 24.0], denominator=[1.0, 4.0])
        assert distinto == pytest.approx((2 / 1 + 4 / 4) / 2)
        assert distinto != pytest.approx(mae([10.0, 20.0], [12.0, 24.0]) / 2.5)

    def test_mase_igual_a_uno_significa_empatar_con_el_naive(self) -> None:
        assert mase([0.0, 0.0], [3.0, -3.0], denominator=3.0) == pytest.approx(1.0)

    def test_un_train_mas_corto_que_la_estacion_no_tiene_denominador(self) -> None:
        assert math.isnan(seasonal_naive_mae([1.0, 2.0], season=2))
        assert math.isnan(seasonal_naive_mae([], season=2))

    def test_los_huecos_del_train_no_contaminan_la_escala(self) -> None:
        # Un hueco elimina **dos** diferencias, no una: la que lo usa como valor
        # actual y la que lo usa como valor retardado. De las cuatro quedan
        # |30-10| = 20 y |50-30| = 20; |NaN-20| y |62-NaN| desaparecen.
        con_hueco = seasonal_naive_mae([10, 20, 30, float("nan"), 50, 62], season=2)
        assert con_hueco == pytest.approx(20.0)

    def test_un_denominador_nulo_deja_la_metrica_indefinida(self) -> None:
        with pytest.warns(UnstableMetricWarning, match="no comete ningun error"):
            assert math.isnan(seasonal_naive_mae([5.0, 5.0, 5.0, 5.0], season=2))
        assert math.isnan(mase(Y, Y_HAT, denominator=0.0))

    def test_rechaza_una_estacionalidad_invalida(self) -> None:
        with pytest.raises(ValueError, match="estacional debe ser >= 1"):
            seasonal_naive_mae([1.0, 2.0, 3.0], season=0)

    def test_rechaza_un_vector_de_denominadores_descuadrado(self) -> None:
        with pytest.raises(ValueError, match="escalar o tener"):
            mase(Y, Y_HAT, denominator=[1.0, 2.0])


class TestMetricasProbabilisticas:
    def test_pinball_penaliza_asimetricamente(self) -> None:
        # tau = 0.25: quedarse corto cuesta 0.25 por unidad, pasarse cuesta 0.75.
        assert pinball_loss([11.0], [8.0], quantile=0.25) == pytest.approx(0.75)
        assert pinball_loss([8.0], [11.0], quantile=0.25) == pytest.approx(2.25)
        # tau = 0.75: al reves.
        assert pinball_loss([11.0], [13.0], quantile=0.75) == pytest.approx(0.5)
        assert pinball_loss([13.0], [11.0], quantile=0.75) == pytest.approx(1.5)

    def test_la_mediana_minimiza_la_pinball_de_0_5(self) -> None:
        muestra = [1.0, 2.0, 3.0, 10.0, 20.0]
        mediana = pinball_loss(muestra, [3.0] * 5, quantile=0.5)
        peor = pinball_loss(muestra, [10.0] * 5, quantile=0.5)
        assert mediana < peor

    def test_pinball_rechaza_cuantiles_fuera_de_rango(self) -> None:
        with pytest.raises(ValueError, match="fuera de"):
            pinball_loss([1.0], [1.0], quantile=1.0)

    def test_cobertura_empirica(self) -> None:
        # dentro, fuera, fuera, dentro
        cobertura = empirical_coverage(
            [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 3.0], [2.0, 1.0, 6.0, 5.0]
        )
        assert cobertura == pytest.approx(0.5)

    def test_la_cobertura_incluye_los_extremos(self) -> None:
        assert empirical_coverage([1.0, 2.0], [1.0, 0.0], [3.0, 2.0]) == pytest.approx(1.0)

    def test_ancho_medio_del_intervalo(self) -> None:
        # (2 + 1 + 1 + 2) / 4
        ancho = interval_width([0.0, 0.0, 5.0, 3.0], [2.0, 1.0, 6.0, 5.0])
        assert ancho == pytest.approx(1.5)

    def test_crps_por_integracion_sobre_la_rejilla(self) -> None:
        # y = 11, cuantiles 0.25/0.5/0.75 en 8/10/13.
        # pinball  = 0.75, 0.50, 0.50
        # 2 * PL   = 1.50, 1.00, 1.00
        # trapecio = 0.25*(1.50+1.00)/2 + 0.25*(1.00+1.00)/2 = 0.3125 + 0.25
        assert crps_discrete(
            [11.0], [[8.0, 10.0, 13.0]], quantiles=(0.25, 0.5, 0.75)
        ) == pytest.approx(0.5625)

    def test_el_crps_premia_a_la_distribucion_mas_ajustada(self) -> None:
        estrecha = crps_discrete([10.0], [[9.0, 10.0, 11.0]], quantiles=(0.25, 0.5, 0.75))
        ancha = crps_discrete([10.0], [[0.0, 10.0, 20.0]], quantiles=(0.25, 0.5, 0.75))
        assert estrecha < ancha

    def test_el_crps_exige_al_menos_dos_cuantiles(self) -> None:
        with pytest.raises(ValueError, match="al menos dos cuantiles"):
            crps_discrete([1.0], [[1.0]], quantiles=(0.5,))

    def test_el_crps_exige_una_rejilla_creciente(self) -> None:
        with pytest.raises(ValueError, match="creciente"):
            crps_discrete([1.0], [[1.0, 2.0]], quantiles=(0.9, 0.1))

    def test_los_intervalos_centrales_de_la_rejilla_canonica(self) -> None:
        niveles = interval_levels((0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975))
        assert [round(level * 100) for level, _, _ in niveles] == [95, 80, 50]
        assert niveles[0] == (0.95, 0.025, 0.975)

    def test_un_cuantil_sin_pareja_no_forma_intervalo(self) -> None:
        assert interval_levels((0.1, 0.5, 0.75)) == ()


class TestEntradasDegeneradas:
    """Sin datos se devuelve `NaN`; con datos descuadrados, un error."""

    def test_las_probabilisticas_sin_observaciones_son_nan(self) -> None:
        assert math.isnan(pinball_loss([], [], quantile=0.5))
        assert math.isnan(empirical_coverage([], [], []))
        assert math.isnan(interval_width([], []))
        assert math.isnan(crps_discrete([float("nan")], [[1.0, 2.0]], quantiles=(0.25, 0.75)))

    @pytest.mark.parametrize(
        "llamada",
        [
            lambda: empirical_coverage([1.0, 2.0], [0.0], [3.0]),
            lambda: interval_width([0.0, 1.0], [3.0]),
            lambda: crps_discrete([1.0, 2.0], [[1.0, 2.0]], quantiles=(0.25, 0.75)),
            lambda: mase([1.0, 2.0], [1.0], denominator=1.0),
        ],
    )
    def test_rechaza_longitudes_descuadradas(self, llamada: object) -> None:
        with pytest.raises(ValueError):
            llamada()  # type: ignore[operator]

    def test_una_trama_sin_columnas_de_cuantil_no_puntua_probabilisticamente(self) -> None:
        frame = pd.DataFrame({"y": Y, "y_hat": Y_HAT})
        metricas = probabilistic_metrics(frame, quantiles=(0.25, 0.75))

        assert metricas["n_obs_prob"] == 0
        assert math.isnan(metricas["pinball_mean"])
        assert math.isnan(metricas["crps_discrete"])


class TestNivelDeTrama:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unique_id": ["s0"] * 4,
                "window_id": [0] * 4,
                "y": Y,
                "y_hat": Y_HAT,
                "mase_denominator": [20.5] * 4,
                "q_2500": [8.0, 16.0, 27.0, 36.0],
                "q_5000": [11.0, 19.0, 31.0, 39.0],
                "q_7500": [14.0, 22.0, 35.0, 43.0],
            }
        )

    def test_point_metrics_reune_las_cinco_puntuales(self) -> None:
        metricas = point_metrics(self._frame())

        assert metricas["n_obs"] == 4
        assert metricas["mae"] == pytest.approx(2.0)
        assert metricas["rmse"] == pytest.approx(2.1213203435596424)
        assert metricas["mape"] == pytest.approx(10.625)
        assert metricas["smape"] == pytest.approx(10.19089726618041)
        assert metricas["mase"] == pytest.approx(2.0 / 20.5)
        assert metricas["mase_denominator"] == pytest.approx(20.5)

    def test_sin_denominador_no_hay_mase(self) -> None:
        metricas = point_metrics(self._frame().drop(columns=["mase_denominator"]))
        assert math.isnan(metricas["mase"])

    def test_probabilistic_metrics_cubre_pinball_cobertura_y_crps(self) -> None:
        metricas = probabilistic_metrics(self._frame(), quantiles=(0.25, 0.5, 0.75))

        assert metricas["n_obs_prob"] == 4
        # y = 10 con q25 = 8: (10-8)*0.25 = 0.5; y = 20 con 16: 1.0;
        # y = 30 con 27: 0.75; y = 40 con 36: 1.0  ->  media 0.8125
        assert metricas["pinball_q_2500"] == pytest.approx(0.8125)
        assert metricas["pinball_mean"] == pytest.approx(
            np.mean(
                [
                    metricas["pinball_q_2500"],
                    metricas["pinball_q_5000"],
                    metricas["pinball_q_7500"],
                ]
            )
        )
        # El intervalo 50 % cubre 10, 20, 30 y 40: los cuatro.
        assert metricas["coverage_50"] == pytest.approx(1.0)
        # anchuras 6, 6, 8, 7  ->  media 6.75
        assert metricas["width_50"] == pytest.approx(6.75)
        assert metricas["crps_discrete"] > 0

    def test_un_modelo_sin_cuantiles_no_contamina_las_probabilisticas(self) -> None:
        frame = self._frame().assign(q_2500=np.nan, q_5000=np.nan, q_7500=np.nan)
        metricas = probabilistic_metrics(frame, quantiles=(0.25, 0.5, 0.75))

        assert metricas["n_obs_prob"] == 0
        assert math.isnan(metricas["crps_discrete"])
        assert math.isnan(metricas["coverage_50"])


class TestDenominadoresPorVentana:
    """`mase_denominators` recorta el panel al train de cada ventana, y solo a el."""

    def _panel(self) -> Panel:
        # y duplica su valor cada paso: las diferencias estacionales a m = 2 son
        # 3, 6, 12, 24, 48, 96..., asi que cada ventana tiene un denominador
        # distinto y facil de sumar a mano.
        values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
        frame = pd.DataFrame(
            {
                "unique_id": ["s0"] * len(values),
                "ds": pd.date_range("2023-01-02", periods=len(values), freq="h"),
                "y": values,
            }
        )
        spec = PanelSpec(dataset_id=DatasetId("mini"), freq="h", seasonalities=(2, 4))
        return Panel(df=frame, spec=spec)

    def _windows(self) -> pd.DataFrame:
        start = pd.Timestamp("2023-01-02")
        return pd.DataFrame(
            {
                "window_id": [0, 1],
                "stage": ["dev", "holdout"],
                "train_start": [start, start],
                "cutoff": [start + pd.Timedelta(hours=4), start + pd.Timedelta(hours=6)],
            }
        )

    def test_cada_ventana_tiene_el_denominador_de_su_propio_train(self) -> None:
        denominadores = mase_denominators(self._panel(), self._windows(), season=2)

        por_ventana = dict(
            zip(
                denominadores["window_id"],
                denominadores["mase_denominator"],
                strict=True,
            )
        )
        # Ventana 0, train = [1, 2, 4, 8, 16]: diferencias 3, 6, 12  ->  7.0
        assert por_ventana[0] == pytest.approx(7.0)
        # Ventana 1, train = [1, 2, 4, 8, 16, 32, 64]: 3, 6, 12, 24, 48  ->  18.6
        assert por_ventana[1] == pytest.approx(18.6)

    def test_el_denominador_no_mira_mas_alla_del_cutoff(self) -> None:
        # Se envenena todo lo posterior al cutoff de la primera ventana. Si el
        # denominador mirase el futuro, el de la ventana 0 cambiaria.
        panel = self._panel()
        windows = self._windows()
        envenenado = panel.df.copy()
        envenenado.loc[envenenado["ds"] > windows.loc[0, "cutoff"], "y"] *= 1_000_000

        limpio = mase_denominators(panel, windows.head(1), season=2)
        sucio = mase_denominators(Panel(df=envenenado, spec=panel.spec), windows.head(1), season=2)

        assert limpio["mase_denominator"].tolist() == sucio["mase_denominator"].tolist()

    def test_usa_la_estacionalidad_del_panel_por_defecto(self) -> None:
        # `mase_season` es la mas corta declarada, aqui 2.
        por_defecto = mase_denominators(self._panel(), self._windows())
        explicita = mase_denominators(self._panel(), self._windows(), season=2)
        assert por_defecto["mase_denominator"].tolist() == explicita["mase_denominator"].tolist()
