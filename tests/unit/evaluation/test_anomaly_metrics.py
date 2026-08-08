"""Metricas de anomalia: correccion sobre casos calculables a mano y el test adversario.

El test que da sentido al modulo es `TestPointAdjustedEsUnFraude`: reproduce el
*point-adjusted F1* que el proyecto rechaza y comprueba que un detector que emite
**ruido puro** obtiene bajo esa regla un F1 alto, mientras que las metricas que
si usamos lo dejan donde le corresponde. Sin ese test, "no usamos point-adjusted"
seria una afirmacion de un documento en vez de una propiedad comprobada.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chronolab.evaluation.anomaly_metrics import (
    METRIC_COLUMNS,
    affiliation_precision_recall,
    auc_pr,
    common_scorable_mask,
    detection_delay,
    evaluate_detector,
    false_alarm_rate,
    point_precision_recall,
    pr_curve,
    range_auc_pr,
    range_precision_recall,
    runs_to_ranges,
    vus_pr,
)
from chronolab.types import DetectorId


def _mask(spec: str) -> np.ndarray:
    """Mascara booleana a partir de una cadena de ceros y unos."""
    return np.array([character == "1" for character in spec], dtype=bool)


def _point_adjusted_f1(predicted: np.ndarray, actual: np.ndarray) -> float:
    """El *point-adjusted F1* que este proyecto rechaza. Vive solo en los tests.

    Si un segmento real recibe **una sola** marca, se da por detectado entero.
    Se implementa aqui, y en ningun sitio de `src/`, para poder demostrar su
    inflacion en lugar de afirmarla.
    """
    adjusted = predicted.astype(bool).copy()
    for start, end in runs_to_ranges(actual):
        if adjusted[start : end + 1].any():
            adjusted[start : end + 1] = True
    _, _, f1 = point_precision_recall(adjusted, actual)
    return f1


class TestRunsToRanges:
    def test_tiradas_maximas(self) -> None:
        assert runs_to_ranges(_mask("0110011")) == [(1, 2), (5, 6)]

    def test_fusion_con_tolerancia(self) -> None:
        assert runs_to_ranges(_mask("110011"), merge_gap=2) == [(0, 5)]
        assert runs_to_ranges(_mask("110011"), merge_gap=1) == [(0, 1), (4, 5)]

    def test_mascara_vacia(self) -> None:
        assert runs_to_ranges(np.zeros(0, dtype=bool)) == []

    def test_merge_gap_negativo_falla(self) -> None:
        with pytest.raises(ValueError, match="merge_gap"):
            runs_to_ranges(_mask("101"), merge_gap=-1)


class TestRangePrecisionRecall:
    def test_solape_parcial_calculado_a_mano(self) -> None:
        # Real [2,5] (4 pasos), predicho [3,4] (2 pasos).
        # recall = solape/longitud_real = 2/4; precision = 2/2.
        report = range_precision_recall(_mask("0001100000"), _mask("0011110000"))
        assert report.recall == pytest.approx(0.5)
        assert report.precision == pytest.approx(1.0)

    def test_la_cardinalidad_penaliza_fragmentar(self) -> None:
        # Un evento de 6 pasos detectado en 3 trozos de 1: el solape es 3/6,
        # pero el factor de cardinalidad lo divide entre los 3 trozos.
        fragmented = range_precision_recall(_mask("101010"), _mask("111111"))
        assert fragmented.recall == pytest.approx(0.5 / 3.0)
        relaxed = range_precision_recall(_mask("101010"), _mask("111111"), cardinality="one")
        assert relaxed.recall == pytest.approx(0.5)

    def test_el_premio_por_existencia_es_la_puerta_del_point_adjusted(self) -> None:
        # Con alpha=1 el recall solo pregunta "he tocado el evento", que es
        # exactamente el vicio que el modulo rechaza. Esta expuesto para poder
        # medir cuanto infla, no para usarlo.
        strict = range_precision_recall(_mask("100000"), _mask("111111"), alpha=0.0)
        inflated = range_precision_recall(_mask("100000"), _mask("111111"), alpha=1.0)
        assert strict.recall == pytest.approx(1.0 / 6.0)
        assert inflated.recall == pytest.approx(1.0)

    def test_el_sesgo_frontal_premia_detectar_pronto(self) -> None:
        # Pesos front sobre un rango de 4: [4,3,2,1], total 10.
        early = range_precision_recall(_mask("1000"), _mask("1111"), bias="front")
        late = range_precision_recall(_mask("0001"), _mask("1111"), bias="front")
        assert early.recall == pytest.approx(0.4)
        assert late.recall == pytest.approx(0.1)

    def test_el_sesgo_plano_no_distingue_posicion(self) -> None:
        early = range_precision_recall(_mask("1000"), _mask("1111"))
        late = range_precision_recall(_mask("0001"), _mask("1111"))
        assert early.recall == pytest.approx(late.recall)

    def test_sin_predicciones_el_recall_es_cero_y_la_precision_indefinida(self) -> None:
        report = range_precision_recall(_mask("0000"), _mask("0110"))
        assert report.recall == pytest.approx(0.0)
        assert math.isnan(report.precision)

    def test_sin_eventos_reales_el_recall_es_indefinido(self) -> None:
        # No es cero: "no habia nada que detectar" y "no detecte nada" son
        # resultados distintos y colapsarlos falsea cualquier promedio.
        report = range_precision_recall(_mask("0110"), _mask("0000"))
        assert math.isnan(report.recall)
        assert report.precision == pytest.approx(0.0)

    def test_las_contribuciones_individuales_permiten_agregar(self) -> None:
        report = range_precision_recall(_mask("110000"), _mask("110011"))
        assert report.per_true_range.size == 2
        assert report.per_true_range.mean() == pytest.approx(report.recall)

    def test_tamanos_distintos_fallan(self) -> None:
        with pytest.raises(ValueError, match="mismo tamano"):
            range_precision_recall(_mask("11"), _mask("111"))


class TestAffiliation:
    def test_deteccion_exacta_de_un_punto(self) -> None:
        # Zona = toda la serie (10 pasos), evento en la posicion 3.
        # precision = |{t : dist(t,evento) > 0}| / 10 = 9/10.
        report = affiliation_precision_recall(_mask("0001000000"), _mask("0001000000"))
        assert report.precision == pytest.approx(0.9)
        assert report.recall == pytest.approx(0.9)

    def test_fallar_por_un_paso_no_es_lo_mismo_que_no_detectar(self) -> None:
        # Es la propiedad que justifica la metrica: con solape estricto ambas
        # valen cero y son indistinguibles.
        near = affiliation_precision_recall(_mask("0010000000"), _mask("0001000000"))
        far = affiliation_precision_recall(_mask("0000000001"), _mask("0001000000"))
        assert near.recall == pytest.approx(0.7)
        assert far.recall < near.recall
        assert range_precision_recall(_mask("0010000000"), _mask("0001000000")).recall == 0.0

    def test_una_zona_sin_predicciones_aporta_recall_cero(self) -> None:
        report = affiliation_precision_recall(_mask("0000000000"), _mask("0001000000"))
        assert report.recall == pytest.approx(0.0)
        assert math.isnan(report.precision)

    def test_el_azar_vale_medio_no_cero(self) -> None:
        # La trampa de lectura de esta metrica: 0.5 no es "aprobado raspado",
        # es ruido. Se comprueba porque es lo que hay que saber para leerla.
        rng = np.random.default_rng(0)
        size = 4000
        actual = np.zeros(size, dtype=bool)
        for start in range(100, size, 200):
            actual[start : start + 5] = True
        predicted = rng.random(size) < 0.05
        report = affiliation_precision_recall(predicted, actual)
        assert report.precision == pytest.approx(0.5, abs=0.05)

    def test_las_zonas_parten_el_hueco_por_la_mitad(self) -> None:
        # Dos eventos: cada punto se afilia al mas cercano, asi que el recall
        # del segundo no puede beneficiarse de una prediccion pegada al primero.
        actual = _mask("11000000000011")
        report = affiliation_precision_recall(_mask("11000000000000"), actual)
        assert report.per_zone_recall[0] > report.per_zone_recall[1]
        assert report.n_zones == 2

    def test_sin_eventos_reales_todo_es_indefinido(self) -> None:
        report = affiliation_precision_recall(_mask("0110"), _mask("0000"))
        assert math.isnan(report.precision)
        assert math.isnan(report.recall)


class TestAucPr:
    def test_precision_media_calculada_a_mano(self) -> None:
        # (0.5-0)*1.0 + (1.0-0.5)*(2/3) = 0.8333...
        report = auc_pr(np.array([0.9, 0.8, 0.7, 0.6]), _mask("1010"))
        assert report.area == pytest.approx(0.5 + 0.5 * 2.0 / 3.0)

    def test_ranking_perfecto(self) -> None:
        report = auc_pr(np.array([0.9, 0.8, 0.2, 0.1]), _mask("1100"))
        assert report.area == pytest.approx(1.0)

    def test_la_linea_base_es_la_prevalencia(self) -> None:
        # Un AUC-PR sin prevalencia al lado no se puede leer: 0.30 es excelente
        # con prevalencia 1 % y pesimo con prevalencia 29 %.
        report = auc_pr(np.array([0.4, 0.3, 0.2, 0.1]), _mask("1000"))
        assert report.baseline == pytest.approx(0.25)

    def test_los_empates_se_agrupan(self) -> None:
        # Con todos los scores iguales solo existe un umbral util, y la
        # precision alcanzable es la prevalencia.
        report = auc_pr(np.array([1.0, 1.0, 1.0, 1.0]), _mask("1010"))
        assert report.area == pytest.approx(0.5)

    def test_los_nan_no_se_marcan_nunca(self) -> None:
        report = auc_pr(np.array([np.nan, 0.8, 0.7, np.nan]), _mask("0110"))
        assert report.area == pytest.approx(1.0)

    def test_los_puntos_sin_score_empatan_en_un_solo_umbral(self) -> None:
        # Entran como -inf, y `-inf - (-inf)` es NaN, que `flatnonzero` cuenta
        # como no nulo: con `numpy.diff` cada uno abriria su propio umbral y la
        # curva pasaria por puntos inalcanzables.
        with np.errstate(invalid="raise"):
            _, recall, thresholds = pr_curve(np.array([0.9, np.nan, np.nan, np.nan]), _mask("1001"))
        assert np.isneginf(thresholds).sum() == 1
        assert recall.size == 3

    def test_sin_positivos_es_indefinido(self) -> None:
        assert math.isnan(auc_pr(np.array([0.5, 0.4]), _mask("00")).area)

    def test_la_curva_arranca_en_precision_uno_y_recall_cero(self) -> None:
        precision, recall, _ = pr_curve(np.array([0.9, 0.1]), _mask("10"))
        assert precision[0] == pytest.approx(1.0)
        assert recall[0] == pytest.approx(0.0)


class TestVusPr:
    def test_tolerar_desalineamiento_recupera_una_deteccion_tardia(self) -> None:
        # Un detector que avisa tres pasos tarde puntua cero con solape
        # estricto; con margen deja de estar penalizado por completo.
        size = 200
        actual = np.zeros(size, dtype=bool)
        actual[100:105] = True
        scores = np.zeros(size)
        scores[108:113] = 1.0
        assert range_auc_pr(scores, actual, buffer=0) < range_auc_pr(scores, actual, buffer=16)

    def test_es_la_media_de_los_cortes(self) -> None:
        rng = np.random.default_rng(3)
        size = 400
        actual = np.zeros(size, dtype=bool)
        actual[50:60] = True
        actual[200:210] = True
        scores = rng.random(size)
        scores[50:60] += 2.0
        expected = np.mean([range_auc_pr(scores, actual, buffer=b) for b in range(0, 9)])
        assert vus_pr(scores, actual, max_buffer=8) == pytest.approx(expected)

    def test_un_detector_perfecto_supera_a_uno_aleatorio(self) -> None:
        rng = np.random.default_rng(5)
        size = 1000
        actual = np.zeros(size, dtype=bool)
        actual[300:320] = True
        perfect = actual.astype(float)
        noise = rng.random(size)
        assert vus_pr(perfect, actual, max_buffer=10) > vus_pr(noise, actual, max_buffer=10)

    def test_sin_positivos_es_indefinido(self) -> None:
        assert math.isnan(vus_pr(np.array([0.5, 0.4]), _mask("00"), max_buffer=2))

    def test_parametros_invalidos_fallan(self) -> None:
        with pytest.raises(ValueError, match="max_buffer"):
            vus_pr(np.array([0.5]), _mask("1"), max_buffer=-1)
        with pytest.raises(ValueError, match="n_buffers"):
            vus_pr(np.array([0.5]), _mask("1"), max_buffer=2, n_buffers=0)


class TestOperativas:
    def test_el_retardo_se_cuenta_desde_el_inicio_del_evento(self) -> None:
        report = detection_delay(_mask("0000110000"), _mask("0011110000"))
        assert report.delays.tolist() == [2.0]
        assert report.mean_delay_steps == pytest.approx(2.0)

    def test_un_evento_no_detectado_no_aporta_retardo(self) -> None:
        # Si aportase, un detector que se pierde lo dificil mejoraria su
        # retardo medio precisamente por perderselo.
        report = detection_delay(_mask("1100000000"), _mask("1100001100"))
        assert report.n_true_events == 2
        assert report.n_detected_events == 1
        assert report.detection_rate == pytest.approx(0.5)
        assert report.delays.size == 1

    def test_las_falsas_alarmas_se_cuentan_por_evento(self) -> None:
        # Cinco marcas seguidas lejos de todo son **una** alarma, no cinco: a
        # un operador se le avisa una vez por incidente.
        report = detection_delay(_mask("0000011111"), _mask("1100000000"))
        assert report.n_false_alarm_events == 1
        assert report.false_alarms_per_1000 == pytest.approx(100.0)

    def test_la_tolerancia_de_fusion_reduce_las_alarmas(self) -> None:
        assert false_alarm_rate(_mask("0010101000"), _mask("1000000000"), merge_gap=0) > (
            false_alarm_rate(_mask("0010101000"), _mask("1000000000"), merge_gap=2)
        )

    def test_per_invalido_falla(self) -> None:
        with pytest.raises(ValueError, match="per"):
            false_alarm_rate(_mask("10"), _mask("10"), per=0)


class TestPointAdjustedEsUnFraude:
    """La razon de existir del modulo, comprobada en vez de afirmada."""

    @staticmethod
    def _ruido() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        size = 10_000
        actual = np.zeros(size, dtype=bool)
        for start in range(200, size, 1000):
            actual[start : start + 50] = True
        return rng.random(size) < 0.05, actual

    def test_el_ruido_aleatorio_saca_un_f1_point_adjusted_alto(self) -> None:
        predicted, actual = self._ruido()
        assert _point_adjusted_f1(predicted, actual) > 0.55

    def test_las_metricas_que_si_usamos_lo_dejan_donde_le_corresponde(self) -> None:
        predicted, actual = self._ruido()
        assert point_precision_recall(predicted, actual)[2] < 0.10
        assert range_precision_recall(predicted, actual).f1 < 0.10

    def test_la_inflacion_es_de_casi_un_orden_de_magnitud(self) -> None:
        predicted, actual = self._ruido()
        honest = point_precision_recall(predicted, actual)[2]
        assert _point_adjusted_f1(predicted, actual) > 5.0 * honest


class TestCommonScorableMask:
    @staticmethod
    def _scores(scorable: list[bool]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unique_id": "s00",
                "ds": pd.date_range("2023-01-01", periods=len(scorable), freq="h"),
                "score": 1.0,
                "scorable": scorable,
            }
        )

    def test_es_la_interseccion(self) -> None:
        mask = common_scorable_mask(
            {
                DetectorId("a"): self._scores([True, True, False]),
                DetectorId("b"): self._scores([True, False, True]),
            }
        )
        assert mask["scorable"].tolist() == [True, False, False]

    def test_una_resolucion_distinta_de_ds_sigue_siendo_la_misma_rejilla(self) -> None:
        # Matrix Profile devolvia `ds` en microsegundos por la union interna, y
        # `DataFrame.equals` compara el dtype: la comparativa entera abortaba
        # con instantes identicos. Lo que se comprueba es la rejilla, no su
        # representacion.
        other = self._scores([True, False, True])
        other["ds"] = other["ds"].astype("datetime64[us]")
        mask = common_scorable_mask(
            {DetectorId("a"): self._scores([True, True, True]), DetectorId("b"): other}
        )
        assert mask["scorable"].tolist() == [True, False, True]

    def test_rejillas_distintas_fallan(self) -> None:
        other = self._scores([True, True])
        other["ds"] = pd.date_range("2024-01-01", periods=2, freq="h")
        with pytest.raises(ValueError, match="rejilla"):
            common_scorable_mask(
                {DetectorId("a"): self._scores([True, True]), DetectorId("b"): other}
            )

    def test_mapa_vacio_falla(self) -> None:
        with pytest.raises(ValueError, match="al menos"):
            common_scorable_mask({})


class TestEvaluateDetector:
    @staticmethod
    def _caso() -> tuple[pd.DataFrame, pd.DataFrame]:
        size = 600
        grid = pd.date_range("2023-01-01", periods=size, freq="h")
        rng = np.random.default_rng(11)
        score = rng.random(size) * 0.5
        anomaly_type = np.full(size, None, dtype=object)
        for start, kind in ((100, "spike"), (300, "level_shift")):
            score[start : start + 10] = 3.0
            anomaly_type[start : start + 10] = kind

        scores = pd.DataFrame({"unique_id": "s00", "ds": grid, "score": score, "scorable": True})
        marked = anomaly_type != None  # noqa: E711  comparacion elemento a elemento
        truth = pd.DataFrame(
            {
                "unique_id": "s00",
                "ds": grid[marked],
                "is_anomaly": True,
                "anomaly_type": anomaly_type[marked],
            }
        )
        return scores, truth

    def test_esquema_de_la_tabla(self) -> None:
        scores, truth = self._caso()
        table = evaluate_detector(scores, truth, detector_id=DetectorId("probe"))
        assert list(table.columns) == list(METRIC_COLUMNS)
        assert set(table["anomaly_type"]) == {"all", "spike", "level_shift"}

    def test_hay_grano_por_serie_y_agregado(self) -> None:
        scores, truth = self._caso()
        table = evaluate_detector(scores, truth, detector_id=DetectorId("probe"))
        assert table["unique_id"].isna().any()
        assert (table["unique_id"] == "s00").any()

    def test_un_detector_perfecto_recupera_ambos_tipos(self) -> None:
        scores, truth = self._caso()
        table = evaluate_detector(scores, truth, detector_id=DetectorId("probe"))
        recall = table.loc[
            (table["metric"] == "range_recall") & table["unique_id"].isna()
        ].set_index("anomaly_type")["value"]
        assert recall["spike"] == pytest.approx(1.0)
        assert recall["level_shift"] == pytest.approx(1.0)

    def test_el_grano_por_tipo_no_emite_precision(self) -> None:
        # Una falsa alarma no pertenece a ningun tipo: cae donde no habia nada.
        # "Precision para escalones" no es una cantidad definida.
        scores, truth = self._caso()
        table = evaluate_detector(scores, truth, detector_id=DetectorId("probe"))
        per_type = table.loc[table["anomaly_type"] != "all", "metric"].unique()
        assert "range_precision" not in per_type
        assert "vus_pr" not in per_type
        assert "false_alarms_per_1000" not in per_type
        assert "range_recall" in per_type

    def test_la_mascara_comun_recorta_el_soporte(self) -> None:
        scores, truth = self._caso()
        support = scores[["unique_id", "ds"]].copy()
        support["scorable"] = False
        support.loc[support.index[:200], "scorable"] = True
        table = evaluate_detector(scores, truth, detector_id=DetectorId("probe"), support=support)
        n_obs = table.loc[table["anomaly_type"] == "all", "n_obs"].max()
        assert n_obs == 200

    def test_alpha_fuera_de_rango_falla(self) -> None:
        scores, truth = self._caso()
        with pytest.raises(ValueError, match="alpha"):
            evaluate_detector(scores, truth, detector_id=DetectorId("probe"), alpha=1.0)

    def test_faltan_columnas(self) -> None:
        scores, truth = self._caso()
        with pytest.raises(ValueError, match="scorable"):
            evaluate_detector(
                scores.drop(columns=["scorable"]), truth, detector_id=DetectorId("probe")
            )
