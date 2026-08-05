"""Diebold-Mariano, correccion de multiplicidad y matriz de pares.

El ejemplo base del contraste se hace a mano de principio a fin:

    perdidas A = [2, 3, 4, 5]      perdidas B = [1, 1, 1, 1]
    d          = [1, 2, 3, 4]      media = 2.5
    gamma_0    = ((-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2) / 4 = 1.25
    V(media)   = gamma_0 / n = 0.3125
    DM         = 2.5 / sqrt(0.3125) = 4.472135955
    HLN(h=1)   = sqrt((4 + 1 - 2) / 4) = sqrt(0.75)
    DM*        = 4.472135955 * 0.866025404 = 3.872983346

Las funciones de distribucion propias se contrastan contra valores de tabla —los
que aparecen en cualquier manual— y, si scipy esta instalado, tambien contra
scipy. Sin ese doble control, un error en la beta incompleta produciria p-valores
plausibles y equivocados, que es la peor clase de error en una tabla que se
publica.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chronolab.errors import UnstableMetricWarning
from chronolab.evaluation.stats_tests import (
    PAIRWISE_COLUMNS,
    _normal_sf,
    _student_t_sf,
    adjust_p_values,
    diebold_mariano,
    dm_from_errors,
    pairwise_dm,
)

LOSS_A = [2.0, 3.0, 4.0, 5.0]
LOSS_B = [1.0, 1.0, 1.0, 1.0]


class TestDistribuciones:
    @pytest.mark.parametrize(
        ("t", "df", "cola"),
        [
            # Valores criticos de tabla: t(0.975, 10) = 2.228, t(0.95, 1) = 6.314,
            # t(0.995, 20) = 2.845.
            (2.228, 10, 0.025),
            (6.314, 1, 0.05),
            (2.845, 20, 0.005),
        ],
    )
    def test_la_t_de_student_reproduce_la_tabla(self, t: float, df: int, cola: float) -> None:
        assert _student_t_sf(t, df=df) == pytest.approx(cola, abs=5e-5)

    def test_la_t_es_simetrica(self) -> None:
        assert _student_t_sf(1.5, df=7) + _student_t_sf(-1.5, df=7) == pytest.approx(1.0)
        assert _student_t_sf(0.0, df=7) == pytest.approx(0.5)

    def test_la_normal_reproduce_la_tabla(self) -> None:
        assert _normal_sf(1.959963985) == pytest.approx(0.025, abs=1e-9)
        assert _normal_sf(1.644853627) == pytest.approx(0.05, abs=1e-9)
        assert _normal_sf(0.0) == pytest.approx(0.5)

    def test_la_t_coincide_con_scipy(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        for t in (0.1, 1.0, 2.5, 4.0, 12.0):
            for df in (1, 3, 10, 47):
                assert _student_t_sf(t, df=df) == pytest.approx(
                    float(stats.t.sf(t, df)), rel=1e-9, abs=1e-12
                )


class TestDieboldMariano:
    def test_el_estadistico_es_el_calculado_a_mano(self) -> None:
        resultado = diebold_mariano(LOSS_A, LOSS_B)
        assert resultado.stat == pytest.approx(3.872983346207417)
        assert resultado.mean_difference == pytest.approx(2.5)
        assert resultado.n_obs == 4
        assert resultado.hln_corrected is True

    def test_la_correccion_hln_encoge_el_estadistico_por_el_factor_exacto(self) -> None:
        sin_correccion = diebold_mariano(LOSS_A, LOSS_B, hln=False)
        con_correccion = diebold_mariano(LOSS_A, LOSS_B, hln=True)

        assert sin_correccion.stat == pytest.approx(4.47213595499958)
        assert con_correccion.stat == pytest.approx(sin_correccion.stat * math.sqrt(0.75))
        assert sin_correccion.hln_corrected is False

    def test_el_p_valor_cae_donde_dice_la_tabla(self) -> None:
        # Con 3 grados de libertad, t = 3.873 esta entre el valor critico del
        # 5 % (3.182) y el del 1 % (5.841): el p-valor bilateral tiene que caer
        # entre 0.01 y 0.05 sin necesidad de creerse la implementacion.
        resultado = diebold_mariano(LOSS_A, LOSS_B)
        assert 0.01 < resultado.p_value < 0.05

    def test_el_p_valor_coincide_con_scipy(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        resultado = diebold_mariano(LOSS_A, LOSS_B)
        esperado = 2.0 * float(stats.t.sf(abs(resultado.stat), 3))
        assert resultado.p_value == pytest.approx(esperado, rel=1e-9)

    def test_el_signo_indica_quien_gana(self) -> None:
        # A pierde mas que B: estadistico positivo. Al invertir, negativo.
        assert diebold_mariano(LOSS_A, LOSS_B).stat > 0
        assert diebold_mariano(LOSS_B, LOSS_A).stat == pytest.approx(
            -diebold_mariano(LOSS_A, LOSS_B).stat
        )

    def test_errores_identicos_no_son_evidencia_de_nada(self) -> None:
        # El caso que hay que tratar a mano: la varianza es cero y el cociente
        # seria 0/0. La respuesta correcta es "no puedo distinguirlos".
        resultado = diebold_mariano([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        assert resultado.stat == 0.0
        assert resultado.p_value == 1.0
        assert resultado.degenerate is True

    def test_un_modelo_comparado_consigo_mismo_empata(self) -> None:
        muestra = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert diebold_mariano(muestra, muestra).p_value == 1.0

    def test_una_diferencia_constante_y_no_nula_diverge_con_aviso(self) -> None:
        with pytest.warns(UnstableMetricWarning, match="constante y no nula"):
            resultado = diebold_mariano([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])

        assert math.isinf(resultado.stat)
        assert resultado.stat > 0
        assert resultado.p_value == 0.0
        assert resultado.degenerate is True

    def test_el_retardo_hac_cambia_la_varianza(self) -> None:
        # Diferencias con autocorrelacion positiva fuerte: al reconocerla, la
        # varianza de la media crece y el estadistico encoge. Ignorarla es lo
        # que hace que un backtest a 24 pasos rechace de mas.
        diferencias = np.array([1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0])
        sin_hac = diebold_mariano(diferencias, np.zeros(8), hac_lag=0)
        con_hac = diebold_mariano(diferencias, np.zeros(8), hac_lag=3)

        assert abs(con_hac.stat) < abs(sin_hac.stat)
        assert con_hac.hac_lag == 3

    def test_con_retardo_cero_la_varianza_es_la_muestral(self) -> None:
        resultado = diebold_mariano(LOSS_A, LOSS_B, hac_lag=0, hln=False)
        # gamma_0 / n con gamma_0 poblacional: 1.25 / 4
        assert resultado.stat == pytest.approx(2.5 / math.sqrt(1.25 / 4))

    def test_avisa_cuando_la_correccion_hln_no_es_aplicable(self) -> None:
        # n = 4 y h = 4 anulan el factor de ajuste.
        with pytest.warns(UnstableMetricWarning, match="no es aplicable"):
            resultado = diebold_mariano([1.0, 5.0, 2.0, 9.0], [0.0, 0.0, 0.0, 0.0], hac_lag=3)
        assert resultado.hln_corrected is False

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"hac_lag": -1}, "hac_lag"),
            ({}, "misma longitud"),
        ],
    )
    def test_rechaza_entradas_invalidas(self, kwargs: dict[str, int], match: str) -> None:
        primero = LOSS_A if match == "hac_lag" else [1.0, 2.0]
        segundo = LOSS_B if match == "hac_lag" else [1.0]
        with pytest.raises(ValueError, match=match):
            diebold_mariano(primero, segundo, **kwargs)

    def test_exige_al_menos_dos_observaciones(self) -> None:
        with pytest.raises(ValueError, match="al menos dos observaciones"):
            diebold_mariano([1.0], [2.0])


class TestDesdeErrores:
    def test_la_perdida_absoluta_usa_el_valor_absoluto_del_error(self) -> None:
        observado = [10.0, 10.0, 10.0, 10.0]
        desde_errores = dm_from_errors(observado, [12.0, 13.0, 14.0, 15.0], [11.0] * 4, loss="abs")
        desde_perdidas = diebold_mariano(LOSS_A, LOSS_B)
        assert desde_errores.stat == pytest.approx(desde_perdidas.stat)

    def test_la_perdida_cuadratica_ordena_distinto(self) -> None:
        observado = [0.0] * 6
        # A falla poco muchas veces; B falla mucho una vez. Bajo perdida
        # absoluta empatan; bajo cuadratica, no.
        a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        b = [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]
        assert dm_from_errors(observado, a, b, loss="abs").mean_difference == pytest.approx(0.0)
        assert dm_from_errors(observado, a, b, loss="sq").mean_difference < 0

    def test_descarta_las_filas_incompletas(self) -> None:
        resultado = dm_from_errors(
            [10.0, 10.0, 10.0, 10.0, float("nan")],
            [12.0, 13.0, 14.0, 15.0, 1.0],
            [11.0, 11.0, 11.0, 11.0, 1.0],
        )
        assert resultado.n_obs == 4

    def test_rechaza_una_perdida_desconocida(self) -> None:
        with pytest.raises(ValueError, match="perdida no admitida"):
            dm_from_errors([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], loss="pinball")  # type: ignore[arg-type]


class TestCorreccionDeMultiplicidad:
    def test_bonferroni_multiplica_por_el_tamano_de_la_familia(self) -> None:
        assert adjust_p_values([0.01, 0.04, 0.5], method="bonferroni").tolist() == pytest.approx(
            [0.03, 0.12, 1.0]
        )

    def test_holm_escalona_y_es_menos_conservador_que_bonferroni(self) -> None:
        # Ordenados: 0.01*3 = 0.03, 0.04*2 = 0.08, 0.5*1 = 0.5
        holm = adjust_p_values([0.01, 0.04, 0.5], method="holm")
        assert holm.tolist() == pytest.approx([0.03, 0.08, 0.5])
        assert (holm <= adjust_p_values([0.01, 0.04, 0.5], method="bonferroni")).all()

    def test_holm_impone_monotonia(self) -> None:
        # Sin monotonia, 0.02*3 = 0.06 quedaria por debajo de 0.03*2 = 0.06 y se
        # podria rechazar uno grande tras no poder rechazar uno mas pequeno.
        ajustados = adjust_p_values([0.02, 0.03, 0.04], method="holm")
        assert list(ajustados) == sorted(ajustados)

    def test_benjamini_hochberg_es_mas_laxo(self) -> None:
        bh = adjust_p_values([0.01, 0.04, 0.5], method="bh")
        holm = adjust_p_values([0.01, 0.04, 0.5], method="holm")
        assert (bh <= holm).all()
        # 0.01*3/1 = 0.03, 0.04*3/2 = 0.06, 0.5*3/3 = 0.5
        assert bh.tolist() == pytest.approx([0.03, 0.06, 0.5])

    def test_none_no_toca_nada(self) -> None:
        assert adjust_p_values([0.01, 0.04], method="none").tolist() == [0.01, 0.04]

    def test_ninguna_correccion_supera_uno(self) -> None:
        for method in ("holm", "bonferroni", "bh"):
            ajustados = adjust_p_values([0.4, 0.6, 0.9], method=method)  # type: ignore[arg-type]
            assert (ajustados <= 1.0).all()

    def test_conserva_el_orden_de_entrada(self) -> None:
        ajustados = adjust_p_values([0.5, 0.01], method="holm")
        assert ajustados[0] > ajustados[1]

    def test_rechaza_un_metodo_desconocido(self) -> None:
        with pytest.raises(ValueError, match="no admitido"):
            adjust_p_values([0.1], method="sidak")  # type: ignore[arg-type]


def _forecasts() -> pd.DataFrame:
    """Predicciones de tres modelos sobre dos series, con calidad decreciente."""
    rng = np.random.default_rng(0)
    parts: list[pd.DataFrame] = []
    for uid in ("s0", "s1"):
        instantes = pd.date_range("2023-01-02", periods=24, freq="h")
        observado = 100.0 + rng.normal(0, 1, 24)
        for model_id, ruido in (("bueno", 0.5), ("regular", 2.0), ("malo", 6.0)):
            parts.append(
                pd.DataFrame(
                    {
                        "unique_id": uid,
                        "model_id": model_id,
                        "window_id": 0,
                        "ds": instantes,
                        "y": observado,
                        "y_hat": observado + rng.normal(0, ruido, 24),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


class TestMatrizDePares:
    def test_cada_par_aparece_una_sola_vez(self) -> None:
        matriz = pairwise_dm(_forecasts())

        assert list(matriz.columns) == list(PAIRWISE_COLUMNS)
        assert len(matriz) == 3  # C(3, 2)
        assert list(zip(matriz["model_a"], matriz["model_b"], strict=True)) == [
            ("bueno", "malo"),
            ("bueno", "regular"),
            ("malo", "regular"),
        ]
        assert (matriz["n_comparisons"] == 3).all()

    def test_el_signo_dice_quien_pierde_menos(self) -> None:
        matriz = pairwise_dm(_forecasts()).set_index(["model_a", "model_b"])
        # "bueno" pierde menos que "malo": diferencia de perdidas negativa.
        assert matriz.loc[("bueno", "malo"), "stat"] < 0
        assert matriz.loc[("bueno", "malo"), "mean_difference"] < 0

    def test_el_p_valor_ajustado_nunca_es_menor_que_el_crudo(self) -> None:
        matriz = pairwise_dm(_forecasts(), method="holm")
        assert (matriz["p_value_adjusted"] >= matriz["p_value"]).all()
        assert (matriz["adjust_method"] == "holm").all()

    def test_la_significacion_se_decide_sobre_el_ajustado(self) -> None:
        matriz = pairwise_dm(_forecasts(), alpha=0.05)
        esperado = matriz["p_value_adjusted"] < 0.05
        assert matriz["significant"].tolist() == esperado.tolist()

    def test_por_series_multiplica_las_comparaciones(self) -> None:
        matriz = pairwise_dm(_forecasts(), by_series=True)

        assert len(matriz) == 6  # 3 pares x 2 series
        assert set(matriz["unique_id"]) == {"s0", "s1"}
        assert (matriz["n_comparisons"] == 6).all()

    def test_agrupando_series_el_identificador_es_nulo(self) -> None:
        matriz = pairwise_dm(_forecasts())
        assert matriz["unique_id"].isna().all()

    def test_dos_modelos_identicos_salen_degenerados(self) -> None:
        frame = _forecasts()
        copia = frame[frame["model_id"] == "bueno"].assign(model_id="clon")
        matriz = pairwise_dm(pd.concat([frame, copia], ignore_index=True))

        par = matriz[(matriz["model_a"] == "bueno") & (matriz["model_b"] == "clon")]
        assert bool(par["degenerate"].item()) is True
        assert par["p_value"].item() == 1.0
        assert par["stat"].item() == 0.0

    def test_solo_compara_los_instantes_que_ambos_tienen(self) -> None:
        frame = _forecasts()
        # "malo" pierde la mitad de sus filas, como si hubiese fallado ventanas.
        recortado = pd.concat(
            [
                frame[frame["model_id"] != "malo"],
                frame[frame["model_id"] == "malo"].head(12),
            ],
            ignore_index=True,
        )
        matriz = pairwise_dm(recortado).set_index(["model_a", "model_b"])

        assert matriz.loc[("bueno", "malo"), "n_obs"] == 12
        assert matriz.loc[("bueno", "regular"), "n_obs"] == 48

    def test_permite_restringir_los_modelos(self) -> None:
        matriz = pairwise_dm(_forecasts(), models=["bueno", "malo"])
        assert len(matriz) == 1

    def test_exige_al_menos_dos_modelos(self) -> None:
        frame = _forecasts()
        with pytest.raises(ValueError, match="al menos dos modelos"):
            pairwise_dm(frame[frame["model_id"] == "bueno"])

    def test_un_par_sin_instantes_comunes_no_produce_fila(self) -> None:
        frame = _forecasts()
        primeros = frame["ds"] < frame["ds"].min() + pd.Timedelta(hours=12)
        sin_solape = pd.concat(
            [
                frame[(frame["model_id"] == "bueno") & primeros],
                frame[(frame["model_id"] == "malo") & ~primeros],
            ],
            ignore_index=True,
        )
        matriz = pairwise_dm(sin_solape)

        # Sin instantes en comun no hay contraste posible, y devolver una fila
        # con n_obs = 0 seria peor que no devolverla.
        assert matriz.empty
        assert list(matriz.columns) == list(PAIRWISE_COLUMNS)

    def test_un_modelo_ausente_en_una_serie_solo_se_compara_donde_esta(self) -> None:
        frame = _forecasts()
        recortado = frame[(frame["model_id"] != "malo") | (frame["unique_id"] == "s0")]
        matriz = pairwise_dm(recortado, by_series=True)

        con_malo = matriz[(matriz["model_a"] == "malo") | (matriz["model_b"] == "malo")]
        assert set(con_malo["unique_id"]) == {"s0"}
        assert len(matriz) == 4  # 3 pares en s0 (tres modelos) y 1 en s1 (dos modelos)


class TestFamiliasVacias:
    def test_ajustar_una_familia_vacia_no_falla(self) -> None:
        assert adjust_p_values([], method="holm").size == 0

    def test_un_contraste_con_perdidas_descuadradas_se_rechaza(self) -> None:
        with pytest.raises(ValueError, match="misma longitud"):
            dm_from_errors([1.0, 2.0], [1.0, 2.0], [1.0])
