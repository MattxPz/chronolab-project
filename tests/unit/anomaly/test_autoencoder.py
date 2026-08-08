"""LSTM-Autoencoder: contrato, comparabilidad y separacion entrenamiento/calibracion."""

from __future__ import annotations

import math

import pytest

from chronolab.anomaly.autoencoder import AutoencoderDetector
from chronolab.anomaly.conformal import SCORE_COLUMNS
from chronolab.anomaly.protocols import AnomalyDetector, FittedDetector
from chronolab.artifacts.reader import scoring_frame
from chronolab.errors import CutoffViolation
from tests.fixtures.anomaly import MODEL, homoscedastic, inject_shift, make_result

pytest.importorskip("torch")

RESULT = make_result(residual=homoscedastic(1.0), n_windows=200, holdout_windows=60, seed=2)
CALIB = scoring_frame(RESULT, model_id=MODEL, stage="dev")
HOLDOUT = scoring_frame(RESULT, model_id=MODEL, stage="holdout")


def _detector(**overrides: object) -> AutoencoderDetector:
    defaults: dict[str, object] = {
        "seq_len": 24,
        "hidden_size": 8,
        "latent_size": 4,
        "epochs": 3,
        "min_calib": 30,
    }
    defaults.update(overrides)
    return AutoencoderDetector(**defaults)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def fitted() -> object:
    return _detector().fit(CALIB)


class TestConformidad:
    def test_satisface_los_protocolos(self, fitted: object) -> None:
        assert isinstance(_detector(), AnomalyDetector)
        assert isinstance(fitted, FittedDetector)

    def test_el_cutoff_es_el_final_de_la_calibracion(self, fitted: object) -> None:
        assert fitted.cutoff == CALIB.end  # type: ignore[attr-defined]

    def test_puntuar_antes_del_cutoff_falla(self, fitted: object) -> None:
        with pytest.raises(CutoffViolation):
            fitted.score(CALIB)  # type: ignore[attr-defined]


class TestScore:
    def test_las_columnas_son_las_del_detector_conformal(self, fitted: object) -> None:
        scores = fitted.score(HOLDOUT)  # type: ignore[attr-defined]
        assert list(scores.columns) == list(SCORE_COLUMNS)

    def test_una_fila_por_punto_de_entrada(self, fitted: object) -> None:
        scores = fitted.score(HOLDOUT)  # type: ignore[attr-defined]
        assert len(scores) == len(HOLDOUT.df)

    def test_el_holdout_completo_es_puntuable_gracias_a_la_cola_de_calibracion(
        self, fitted: object
    ) -> None:
        scores = fitted.score(HOLDOUT)  # type: ignore[attr-defined]
        assert bool(scores["scorable"].all())

    def test_score_es_no_negativo_y_satura_en_log10_n_mas_1(self, fitted: object) -> None:
        scores = fitted.score(HOLDOUT)  # type: ignore[attr-defined]
        usable = scores.loc[scores["scorable"], "score"]
        assert (usable >= 0).all()
        calib_n = int(scores.loc[scores["scorable"], "calib_n"].iloc[0])
        assert (usable <= math.log10(calib_n + 1) + 1e-6).all()


class TestComparabilidad:
    def test_un_desplazamiento_grande_sube_el_score(self, fitted: object) -> None:
        uid = str(RESULT.forecasts["unique_id"].iloc[0])
        shifted = inject_shift(RESULT, uid=uid, start=HOLDOUT.start, length=20, magnitude=25.0)
        holdout_shifted = scoring_frame(shifted, model_id=MODEL, stage="holdout")

        baseline = fitted.score(HOLDOUT)  # type: ignore[attr-defined]
        anomalous = fitted.score(holdout_shifted)  # type: ignore[attr-defined]

        mask = (anomalous["unique_id"] == uid) & anomalous["scorable"]
        window_mask = mask & anomalous["ds"].isin(baseline.loc[mask, "ds"].iloc[:20])
        assert anomalous.loc[window_mask, "score"].mean() > baseline.loc[mask, "score"].mean()


class TestValidacion:
    def test_seq_len_menor_que_dos_falla(self) -> None:
        with pytest.raises(ValueError):
            AutoencoderDetector(seq_len=1)

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
            _detector().fit(bad)

    def test_calibracion_insuficiente_falla_al_ajustar(self) -> None:
        with pytest.raises(ValueError):
            _detector(min_calib=10_000).fit(CALIB)
