"""Protocolos de deteccion de anomalias: conformidad y separacion score/etiqueta."""

from __future__ import annotations

import pandas as pd
import pytest

from chronolab.anomaly.protocols import (
    AnomalyDetector,
    DetectorRequirements,
    FittedDetector,
    ScoringFrame,
)
from chronolab.panel import PanelSpec
from chronolab.types import DetectorId


class DummyFittedDetector:
    """Detector calibrado minimo que satisface el protocolo."""

    def __init__(self, cutoff: pd.Timestamp) -> None:
        self._cutoff = cutoff

    @property
    def detector_id(self) -> DetectorId:
        return DetectorId("dummy")

    @property
    def cutoff(self) -> pd.Timestamp:
        return self._cutoff

    def score(self, frame: ScoringFrame) -> pd.DataFrame:
        return pd.DataFrame(columns=["unique_id", "ds", "score", "scorable"])


class DummyDetector:
    """Configuracion minima que satisface el protocolo."""

    @property
    def detector_id(self) -> DetectorId:
        return DetectorId("dummy")

    @property
    def requires(self) -> DetectorRequirements:
        return DetectorRequirements()

    def fit(self, calib: ScoringFrame) -> DummyFittedDetector:
        return DummyFittedDetector(cutoff=calib.end)


def _scoring_frame(spec: PanelSpec) -> ScoringFrame:
    return ScoringFrame(
        df=pd.DataFrame(columns=["unique_id", "ds", "y", "y_hat"]),
        spec=spec,
        model_id=None,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-02-01"),
    )


class TestConformidad:
    def test_detector_estructural(self) -> None:
        assert isinstance(DummyDetector(), AnomalyDetector)

    def test_fitted_detector_estructural(self) -> None:
        assert isinstance(DummyFittedDetector(pd.Timestamp("2023-01-01")), FittedDetector)

    def test_el_cutoff_del_detector_es_el_final_de_la_calibracion(self, spec: PanelSpec) -> None:
        # Es la barrera que impide que el tramo de calibracion se solape con el
        # que se puntua (docs/ARCHITECTURE.md fuga L9).
        calib = _scoring_frame(spec)
        assert DummyDetector().fit(calib).cutoff == calib.end


class TestContratoDeScore:
    def test_score_no_devuelve_etiquetas(self, spec: PanelSpec) -> None:
        # Puntuar y umbralizar estan separados a proposito: VUS-PR necesita el
        # score continuo y F1 por rangos necesita etiquetas. Si el detector
        # devolviera etiquetas se perderia lo que exige la metrica principal.
        detector = DummyDetector().fit(_scoring_frame(spec))
        columns = list(detector.score(_scoring_frame(spec)).columns)
        assert "score" in columns
        assert "scorable" in columns
        assert not any(name in columns for name in ("is_anomaly", "label", "label_pred"))


class TestScoringFrame:
    def test_es_inmutable(self, spec: PanelSpec) -> None:
        with pytest.raises(AttributeError):
            _scoring_frame(spec).model_id = None  # type: ignore[misc]

    def test_admite_detectores_que_no_usan_prediccion(self, spec: PanelSpec) -> None:
        # Matrix Profile no necesita `y_hat`, pero se evalua sobre exactamente
        # los mismos instantes que los demas para que la comparativa signifique
        # algo.
        assert _scoring_frame(spec).model_id is None


class TestDetectorRequirements:
    def test_los_valores_por_defecto_son_los_conservadores(self) -> None:
        requirements = DetectorRequirements()
        assert requirements.needs_forecast is False
        assert requirements.needs_quantiles is False
        assert requirements.window == 1
        assert requirements.needs_calibration is True

    def test_la_ventana_determina_el_calentamiento(self) -> None:
        # Con ventana 512 los primeros 511 puntos no son puntuables, y el
        # evaluador debe igualar mascaras antes de comparar detectores.
        assert DetectorRequirements(window=512).window - 1 == 511
