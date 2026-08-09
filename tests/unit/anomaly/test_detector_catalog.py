"""Todos los `AnomalyDetector`: mismo tamano de score, sin `NaN` donde es puntuable.

Cada detector ya tiene su propio fichero de conformidad
(`test_conformal.py`, `test_isolation.py`, `test_matrix_profile.py`,
`test_autoencoder.py`), y cada uno comprueba por separado que su propio score
no tiene NaN espurio. Lo que ninguno comprueba es la propiedad transversal que
pide docs/ARCHITECTURE.md: que **todos**, puestos a puntuar el mismo tramo,
devuelven exactamente una fila por observacion de entrada, con `scorable`
booleano completo y ningun `NaN` en `score` alli donde `scorable` es
verdadero. Es el equivalente, para detectores, del catalogo transversal de
`Forecaster` en `tests/unit/models/test_forecaster_catalog.py`.

Ninguno de los cuatro detectores ajusta de verdad la CPU tanto como los
adaptadores de modelos mas pesados (ni siquiera `AutoencoderDetector`, que
entrena tres epocas de una red diminuta), asi que este fichero corre completo
en `make test-fast`, igual que sus cuatro ficheros dedicados.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from chronolab.anomaly.conformal import ConformalDetector
from chronolab.anomaly.protocols import AnomalyDetector, FittedDetector
from chronolab.artifacts.reader import scoring_frame
from chronolab.errors import CutoffViolation
from tests.fixtures.anomaly import MODEL, homoscedastic, make_result

MIN_COLUMNS = {"unique_id", "ds", "score", "scorable"}
"""Contrato minimo comun: las columnas extra (`severity`, `calib_n`, `side`)
solo las declara `ConformalDetector`, y no forman parte de lo transversal."""


def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in modules)


def _needs_extra(*modules: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not _installed(*modules), reason=f"requiere {', '.join(modules)}, no instalado"
    )


RESULT = make_result(residual=homoscedastic(1.0), n_windows=200, holdout_windows=60, seed=42)
CALIB = scoring_frame(RESULT, model_id=MODEL, stage="dev")
HOLDOUT = scoring_frame(RESULT, model_id=MODEL, stage="holdout")


def _conformal() -> AnomalyDetector:
    return ConformalDetector(
        base_model_id=MODEL, hour_bins=1, min_calib=200, gamma=0.02, pool_size=250
    )


def _isolation_forest() -> AnomalyDetector:
    from chronolab.anomaly.isolation import IsolationForestDetector

    return IsolationForestDetector(base_model_id=MODEL, window=24, min_calib=30)


def _matrix_profile() -> AnomalyDetector:
    from chronolab.anomaly.matrix_profile import MatrixProfileDetector

    return MatrixProfileDetector(m=24)


def _autoencoder() -> AnomalyDetector:
    from chronolab.anomaly.autoencoder import AutoencoderDetector

    return AutoencoderDetector(seq_len=24, hidden_size=8, latent_size=4, epochs=3, min_calib=30)


CATALOG: list[pytest.param] = [
    pytest.param(_conformal, id="conformal"),
    pytest.param(_isolation_forest, id="isolation_forest", marks=_needs_extra("pyod")),
    pytest.param(_matrix_profile, id="matrix_profile", marks=_needs_extra("stumpy")),
    pytest.param(_autoencoder, id="autoencoder", marks=_needs_extra("torch")),
]


@pytest.fixture(params=CATALOG, scope="module")
def detector(request: pytest.FixtureRequest) -> AnomalyDetector:
    factory: Callable[[], AnomalyDetector] = request.param
    built = factory()
    assert isinstance(built, AnomalyDetector)
    return built


@pytest.fixture(scope="module")
def fitted(detector: AnomalyDetector) -> FittedDetector:
    # Modulo, no funcion: calibrar de verdad (torch incluido) una vez por
    # detector en vez de una vez por test es lo que mantiene este fichero
    # fuera de `slow` a pesar de cubrir cuatro implementaciones.
    fitted_detector = detector.fit(CALIB)
    assert isinstance(fitted_detector, FittedDetector)
    return fitted_detector


class TestConformidad:
    def test_el_cutoff_es_el_final_de_la_calibracion(self, fitted: FittedDetector) -> None:
        assert fitted.cutoff == CALIB.end

    def test_puntuar_el_propio_tramo_de_calibracion_falla(self, fitted: FittedDetector) -> None:
        with pytest.raises(CutoffViolation):
            fitted.score(CALIB)


@pytest.fixture(scope="module")
def scores(fitted: FittedDetector) -> pd.DataFrame:
    """`score(HOLDOUT)`, calculado una vez por detector y reutilizado.

    Las seis comprobaciones de `TestContratoDeScore` son de solo lectura: nada
    en ellas depende de invocar `score` mas de una vez.
    """
    return fitted.score(HOLDOUT)


class TestContratoDeScore:
    def test_una_fila_por_observacion_de_entrada(self, scores: pd.DataFrame) -> None:
        assert len(scores) == len(HOLDOUT.df)

    def test_las_filas_se_alinean_con_la_entrada_en_el_mismo_orden(
        self, scores: pd.DataFrame
    ) -> None:
        assert (
            scores["unique_id"].astype(str).tolist() == HOLDOUT.df["unique_id"].astype(str).tolist()
        )
        pd.testing.assert_series_equal(
            scores["ds"].reset_index(drop=True),
            HOLDOUT.df["ds"].reset_index(drop=True),
            check_dtype=False,
            check_names=False,
        )

    def test_las_columnas_minimas_del_contrato_estan_presentes(self, scores: pd.DataFrame) -> None:
        assert MIN_COLUMNS.issubset(scores.columns)
        # Puntuar y umbralizar estan separados a proposito (docs/ARCHITECTURE.md):
        # ningun detector devuelve ya una etiqueta.
        assert not any(name in scores.columns for name in ("is_anomaly", "label", "label_pred"))

    def test_scorable_es_booleano_y_no_tiene_huecos(self, scores: pd.DataFrame) -> None:
        assert scores["scorable"].dtype == bool
        assert scores["scorable"].notna().all()

    def test_donde_es_puntuable_el_score_no_tiene_nan_ni_infinitos(
        self, scores: pd.DataFrame
    ) -> None:
        usable = scores.loc[scores["scorable"], "score"]
        # Si nada resultase puntuable el resto de este test pasaria sin comprobar
        # nada: con `HOLDOUT` de 60 ventanas y un calentamiento maximo de 24
        # pasos, siempre queda un tramo puntuable de sobra.
        assert not usable.empty
        assert usable.notna().all()
        assert np.isfinite(usable.to_numpy(dtype=float)).all()

    def test_donde_no_es_puntuable_el_score_es_nulo_no_ausente(self, scores: pd.DataFrame) -> None:
        # La fila sigue existiendo (I3): lo que falta es el valor, no la fila.
        unusable = scores.loc[~scores["scorable"], "score"]
        assert unusable.isna().all()
