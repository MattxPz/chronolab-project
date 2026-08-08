"""Matrix Profile: contrato minimo, sin calibracion, y sensibilidad a discords."""

from __future__ import annotations

import pytest

from chronolab.anomaly.matrix_profile import SCORE_COLUMNS, MatrixProfileDetector
from chronolab.anomaly.protocols import AnomalyDetector, FittedDetector
from chronolab.artifacts.reader import scoring_frame
from chronolab.errors import CutoffViolation
from tests.fixtures.anomaly import MODEL, homoscedastic, inject_shift, make_result

pytest.importorskip("stumpy")

RESULT = make_result(residual=homoscedastic(1.0), n_windows=200, holdout_windows=60, seed=4)
CALIB = scoring_frame(RESULT, model_id=MODEL, stage="dev")
HOLDOUT = scoring_frame(RESULT, model_id=MODEL, stage="holdout")


class TestConformidad:
    def test_satisface_los_protocolos(self) -> None:
        detector = MatrixProfileDetector(m=24)
        assert isinstance(detector, AnomalyDetector)
        fitted = detector.fit(CALIB)
        assert isinstance(fitted, FittedDetector)

    def test_no_exige_calibracion(self) -> None:
        assert MatrixProfileDetector(m=24).requires.needs_calibration is False

    def test_el_cutoff_es_el_final_de_la_calibracion(self) -> None:
        assert MatrixProfileDetector(m=24).fit(CALIB).cutoff == CALIB.end

    def test_puntuar_antes_del_cutoff_falla(self) -> None:
        fitted = MatrixProfileDetector(m=24).fit(CALIB)
        with pytest.raises(CutoffViolation):
            fitted.score(CALIB)


class TestScore:
    def test_no_emite_severity_calib_n_ni_side(self) -> None:
        scores = MatrixProfileDetector(m=24).fit(CALIB).score(HOLDOUT)
        assert list(scores.columns) == list(SCORE_COLUMNS)
        assert "severity" not in scores.columns
        assert "side" not in scores.columns

    def test_una_fila_por_punto_de_entrada(self) -> None:
        scores = MatrixProfileDetector(m=24).fit(CALIB).score(HOLDOUT)
        assert len(scores) == len(HOLDOUT.df)

    def test_score_no_negativo(self) -> None:
        scores = MatrixProfileDetector(m=24).fit(CALIB).score(HOLDOUT)
        usable = scores.loc[scores["scorable"], "score"]
        assert (usable >= -1e-6).all()


class TestComparabilidad:
    def test_un_desplazamiento_grande_sube_el_score(self) -> None:
        uid = str(RESULT.forecasts["unique_id"].iloc[0])
        shifted = inject_shift(RESULT, uid=uid, start=HOLDOUT.start, length=20, magnitude=25.0)
        holdout_shifted = scoring_frame(shifted, model_id=MODEL, stage="holdout")

        fitted = MatrixProfileDetector(m=24).fit(CALIB)
        baseline = fitted.score(HOLDOUT)
        anomalous = fitted.score(holdout_shifted)

        mask = (anomalous["unique_id"] == uid) & anomalous["scorable"]
        window_mask = mask & anomalous["ds"].isin(baseline.loc[mask, "ds"].iloc[:20])
        assert anomalous.loc[window_mask, "score"].mean() > baseline.loc[mask, "score"].mean()


class TestValidacion:
    def test_m_menor_que_tres_falla(self) -> None:
        with pytest.raises(ValueError):
            MatrixProfileDetector(m=2)

    def test_falta_la_columna_objetivo(self) -> None:
        from chronolab.anomaly.protocols import ScoringFrame

        bad = ScoringFrame(
            df=CALIB.df.drop(columns=["y"]),
            spec=CALIB.spec,
            model_id=CALIB.model_id,
            start=CALIB.start,
            end=CALIB.end,
        )
        with pytest.raises(ValueError, match="'y'"):
            MatrixProfileDetector(m=24).fit(bad)
