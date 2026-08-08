"""IsolationForest sobre features de ventana: contrato, comparabilidad y split conformal."""

from __future__ import annotations

import math

import pytest

from chronolab.anomaly.conformal import SCORE_COLUMNS
from chronolab.anomaly.isolation import IsolationForestDetector, pool_score
from chronolab.anomaly.protocols import AnomalyDetector, FittedDetector
from chronolab.artifacts.reader import scoring_frame
from chronolab.errors import CutoffViolation
from tests.fixtures.anomaly import MODEL, homoscedastic, inject_shift, make_result

pytest.importorskip("pyod")

RESULT = make_result(residual=homoscedastic(1.0), n_windows=200, holdout_windows=60, seed=1)
CALIB = scoring_frame(RESULT, model_id=MODEL, stage="dev")
HOLDOUT = scoring_frame(RESULT, model_id=MODEL, stage="holdout")


def _detector(**overrides: object) -> IsolationForestDetector:
    defaults: dict[str, object] = {"base_model_id": MODEL, "window": 24, "min_calib": 30}
    defaults.update(overrides)
    return IsolationForestDetector(**defaults)  # type: ignore[arg-type]


class TestConformidad:
    def test_satisface_los_protocolos(self) -> None:
        detector = _detector()
        assert isinstance(detector, AnomalyDetector)
        fitted = detector.fit(CALIB)
        assert isinstance(fitted, FittedDetector)

    def test_el_cutoff_es_el_final_de_la_calibracion(self) -> None:
        assert _detector().fit(CALIB).cutoff == CALIB.end

    def test_puntuar_antes_del_cutoff_falla(self) -> None:
        fitted = _detector().fit(CALIB)
        with pytest.raises(CutoffViolation):
            fitted.score(CALIB)


class TestScore:
    def test_las_columnas_son_las_del_detector_conformal(self) -> None:
        scores = _detector().fit(CALIB).score(HOLDOUT)
        assert list(scores.columns) == list(SCORE_COLUMNS)

    def test_una_fila_por_punto_de_entrada(self) -> None:
        scores = _detector().fit(CALIB).score(HOLDOUT)
        assert len(scores) == len(HOLDOUT.df)

    def test_el_calentamiento_del_holdout_se_puentea_con_la_cola_de_calibracion(self) -> None:
        # `requires.window` cuenta el contexto que el detector consume, pero la
        # cola de `calib` deberia bastar para que el holdout entero sea
        # puntuable sin repetir el calentamiento.
        scores = _detector().fit(CALIB).score(HOLDOUT)
        assert bool(scores["scorable"].all())

    def test_score_es_no_negativo_y_satura_en_log10_n_mas_1(self) -> None:
        fitted = _detector().fit(CALIB)
        scores = fitted.score(HOLDOUT)
        usable = scores.loc[scores["scorable"], "score"]
        assert (usable >= 0).all()
        calib_n = int(scores.loc[scores["scorable"], "calib_n"].iloc[0])
        assert (usable <= math.log10(calib_n + 1) + 1e-6).all()

    def test_side_es_el_signo_del_residuo(self) -> None:
        scores = _detector().fit(CALIB).score(HOLDOUT)
        assert set(scores.loc[scores["scorable"], "side"].unique()) <= {-1, 0, 1}


class TestComparabilidad:
    def test_un_desplazamiento_grande_sube_el_score(self) -> None:
        # Es la propiedad que el usuario pide: el score tiene que ser capaz
        # de senalar una anomalia real, no solo tener un rango acotado.
        uid = str(RESULT.forecasts["unique_id"].iloc[0])
        shifted = inject_shift(RESULT, uid=uid, start=HOLDOUT.start, length=20, magnitude=25.0)
        holdout_shifted = scoring_frame(shifted, model_id=MODEL, stage="holdout")

        fitted = _detector().fit(CALIB)
        baseline = fitted.score(HOLDOUT)
        anomalous = fitted.score(holdout_shifted)

        mask = (anomalous["unique_id"] == uid) & anomalous["scorable"]
        window_mask = mask & anomalous["ds"].isin(baseline.loc[mask, "ds"].iloc[:20])
        shifted_score = anomalous.loc[window_mask, "score"].mean()
        normal_score = baseline.loc[mask, "score"].mean()
        assert shifted_score > normal_score

    def test_pool_score_es_monotona_en_la_magnitud_cruda(self) -> None:
        import numpy as np

        pool = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        low, high = pool_score(np.array([1.5, 4.5]), pool)
        assert high > low


class TestValidacion:
    def test_ventana_menor_que_tres_falla(self) -> None:
        with pytest.raises(ValueError):
            IsolationForestDetector(window=2)

    def test_calib_fraction_fuera_de_rango_falla(self) -> None:
        with pytest.raises(ValueError):
            IsolationForestDetector(calib_fraction=1.0)

    def test_calibracion_insuficiente_falla_al_ajustar(self) -> None:
        with pytest.raises(ValueError):
            IsolationForestDetector(min_calib=10_000).fit(CALIB)
