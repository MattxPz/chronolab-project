"""Detector conformal: contrato, cobertura, heterocedasticidad, deriva y absorcion.

Cuatro de estos tests no comprueban codigo sino **afirmaciones del diseno**. Si
el de heterocedasticidad falla, la calibracion de Mondrian no hace falta; si el
de deriva falla, el modo adaptativo no aporta nada; si el de absorcion falla, la
cuarentena sobra. Estan escritos para poder borrarse si dejan de ser ciertos.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chronolab.anomaly.conformal import ALPHA_GRID, SCORE_COLUMNS, ConformalDetector, _invert
from chronolab.anomaly.protocols import AnomalyDetector, FittedDetector, ScoringFrame
from chronolab.artifacts.reader import scoring_frame
from chronolab.data.calendar import local_hour
from chronolab.errors import CutoffViolation
from tests.fixtures.anomaly import (
    MODEL,
    drifting,
    drop_window,
    homoscedastic,
    hour_dependent,
    inject_shift,
    make_result,
)

ALPHA = 0.05
THRESHOLD = -math.log10(ALPHA)
TZ = "Europe/Madrid"


def _frames(result: object) -> tuple[ScoringFrame, ScoringFrame]:
    return (
        scoring_frame(result, model_id=MODEL, stage="dev"),  # type: ignore[arg-type]
        scoring_frame(result, model_id=MODEL, stage="holdout"),  # type: ignore[arg-type]
    )


def _flagged(scores: pd.DataFrame, alpha: float = ALPHA) -> pd.Series:
    return scores["scorable"] & (scores["score"] >= -math.log10(alpha))


@pytest.fixture(scope="module")
def plain() -> tuple[ScoringFrame, ScoringFrame]:
    """Run homoscedastico, sin anomalias: el caso en el que la teoria aplica."""
    return _frames(make_result(residual=homoscedastic(1.0), n_windows=200, holdout_windows=60))


@pytest.fixture(scope="module")
def heteroscedastic() -> tuple[ScoringFrame, ScoringFrame]:
    """Run con varianza dependiente de la hora local, como la demanda electrica."""
    return _frames(
        make_result(residual=hour_dependent(), n_windows=200, holdout_windows=40, seed=7)
    )


@pytest.fixture(scope="module")
def drifted() -> tuple[ScoringFrame, ScoringFrame]:
    """Run cuya varianza se dispara justo despues de la calibracion."""
    return _frames(
        make_result(
            residual=drifting(0.6, 3.0, onset=0.7),
            n_windows=200,
            holdout_windows=60,
            seed=11,
        )
    )


def _split(**kwargs: object) -> ConformalDetector:
    """Split conformal exacto: sin pool rodante y sin ACI."""
    defaults: dict[str, object] = {
        "base_model_id": MODEL,
        "hour_bins": 6,
        "min_calib": 50,
        "gamma": 0.0,
        "pool_size": None,
    }
    return ConformalDetector(**{**defaults, **kwargs})  # type: ignore[arg-type]


def _adaptive(**kwargs: object) -> ConformalDetector:
    """Adaptativo en linea: pool rodante mas recursion de ACI, con grupos gruesos."""
    defaults: dict[str, object] = {
        "base_model_id": MODEL,
        "hour_bins": 1,
        "min_calib": 200,
        "gamma": 0.02,
        "pool_size": 250,
    }
    return ConformalDetector(**{**defaults, **kwargs})  # type: ignore[arg-type]


class TestConformidad:
    def test_satisface_los_protocolos(self, plain: tuple[ScoringFrame, ScoringFrame]) -> None:
        calib, _ = plain
        detector = _split()
        assert isinstance(detector, AnomalyDetector)
        assert isinstance(detector.fit(calib), FittedDetector)

    def test_declara_que_necesita_cuantiles(self) -> None:
        # No fabrica un intervalo que el modelo no dio: un modelo puntual se
        # envuelve antes con ConformalWrapper.
        requires = _split().requires
        assert requires.needs_forecast is True
        assert requires.needs_quantiles is True
        assert requires.window == 1

    def test_el_cutoff_es_el_final_de_la_calibracion(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        calib, _ = plain
        assert _split().fit(calib).cutoff == calib.end

    def test_el_identificador_distingue_el_metodo_y_el_modelo_base(self) -> None:
        assert _split().detector_id != _split(gamma=0.02).detector_id
        assert _split().detector_id != _split(pool_size=500).detector_id
        assert _split().detector_id != _split(base_model_id=None).detector_id


class TestContratoDeScore:
    def test_devuelve_una_fila_por_punto_y_en_el_mismo_orden(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        calib, test = plain
        scores = _split().fit(calib).score(test)
        assert list(scores.columns) == list(SCORE_COLUMNS)
        assert len(scores) == len(test.df)
        assert scores["ds"].equals(test.df["ds"].reset_index(drop=True))
        assert scores["unique_id"].tolist() == test.df["unique_id"].astype(str).tolist()

    def test_no_devuelve_etiquetas(self, plain: tuple[ScoringFrame, ScoringFrame]) -> None:
        calib, test = plain
        columns = _split().fit(calib).score(test).columns
        assert not any(name in columns for name in ("is_anomaly", "label", "label_pred"))

    def test_donde_no_es_puntuable_el_score_es_nulo(self) -> None:
        # Una ventana fallida no deja filas en `forecasts`. El lector la devuelve
        # como NaN explicito y el detector tiene que marcarla, no saltarsela: si
        # desapareciera, la rejilla dejaria de estar alineada entre detectores.
        result = make_result(residual=homoscedastic(1.0), n_windows=120, holdout_windows=40)
        broken_id = int(
            result.windows.loc[result.windows["stage"] == "holdout", "window_id"].iloc[3]
        )
        calib, test = _frames(drop_window(result, window_id=broken_id))

        scores = _split().fit(calib).score(test)
        hole = test.df["y"].isna().to_numpy()

        assert hole.any()
        assert len(scores) == len(test.df)
        assert not scores.loc[hole, "scorable"].any()
        assert scores.loc[hole, "score"].isna().all()
        assert (scores.loc[hole, "calib_n"] == 0).all()
        assert (scores.loc[hole, "side"] == 0).all()

    def test_el_score_satura_en_el_tamano_del_grupo(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        # Con n puntos de calibracion el p-valor no baja de 1/(n+1): la cola por
        # encima de eso no se ha observado y no se extrapola.
        calib, test = plain
        scores = _split().fit(calib).score(test)
        usable = scores.loc[scores["scorable"]]
        ceiling = np.log10(usable["calib_n"].to_numpy(dtype=float) + 1.0)
        assert (usable["score"].to_numpy(dtype=float) <= ceiling + 1e-6).all()


class TestBarreraDeCutoff:
    def test_puntuar_antes_del_cutoff_es_fuga(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        calib, _ = plain
        fitted = _split().fit(calib)
        with pytest.raises(CutoffViolation):
            fitted.score(calib)

    def test_avanzar_antes_del_cutoff_es_fuga(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        calib, _ = plain
        fitted = _split().fit(calib)
        with pytest.raises(CutoffViolation):
            fitted.advance(calib)


class TestDeterminismo:
    def test_puntuar_dos_veces_da_lo_mismo(self, plain: tuple[ScoringFrame, ScoringFrame]) -> None:
        # El detector es secuencial por dentro; si `score` mutase el estado, el
        # segundo resultado seria otro y nada lo diria.
        calib, test = plain
        fitted = _adaptive().fit(calib)
        pd.testing.assert_frame_equal(fitted.score(test), fitted.score(test))

    def test_puntuar_no_avanza_el_estado(self, plain: tuple[ScoringFrame, ScoringFrame]) -> None:
        calib, test = plain
        fitted = _adaptive().fit(calib)
        before = fitted.coverage_report()
        fitted.score(test)
        pd.testing.assert_frame_equal(before, fitted.coverage_report())

    def test_avanzar_devuelve_un_detector_nuevo(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        calib, test = plain
        fitted = _adaptive().fit(calib)
        advanced = fitted.advance(test)
        assert advanced is not fitted
        assert fitted.cutoff == calib.end
        assert advanced.cutoff == test.end
        assert (
            advanced.coverage_report()["n_scored"].sum()
            > fitted.coverage_report()["n_scored"].sum()
        )


class TestCobertura:
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_la_tasa_de_falsos_positivos_queda_acotada(
        self, plain: tuple[ScoringFrame, ScoringFrame], alpha: float
    ) -> None:
        # Es la afirmacion central: sobre datos intercambiables la tasa de
        # marcado no pasa de alpha, y no se ha asumido normalidad en ningun sitio.
        calib, test = plain
        scores = _split().fit(calib).score(test)
        rate = float(_flagged(scores, alpha).mean())
        assert rate <= alpha * 1.5

    def test_severity_y_score_no_pueden_discrepar(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        # severity = (r - Q) / (1 + 2Q) es una transformacion afin creciente de la
        # no conformidad, asi que marcar por una o por otra es la misma decision.
        calib, test = plain
        scores = _split().fit(calib).score(test)
        usable = scores.loc[scores["scorable"]]
        by_score = usable["score"] >= THRESHOLD
        by_severity = usable["severity"] > 0
        assert by_score.equals(by_severity)


class TestHeteroscedasticidad:
    def test_un_umbral_global_falla_por_hora_y_mondrian_no(
        self, heteroscedastic: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        # Esta es la prueba que justifica la calibracion por grupos. El detector
        # global sale con una tasa marginal correcta —parece calibrado— y una
        # tasa condicional desastrosa, que es exactamente el error que un umbral
        # global produce sin dejar sintoma.
        calib, test = heteroscedastic

        def by_hour(detector: ConformalDetector) -> pd.Series:
            scores = detector.fit(calib).score(test)
            hour = local_hour(scores["ds"], tz_display=TZ)
            return _flagged(scores).groupby(hour).mean()

        globally = by_hour(_split(hour_bins=1, min_calib=200))
        mondrian = by_hour(_split(hour_bins=24, min_calib=50))

        assert abs(float(globally.mean()) - ALPHA) < 0.02
        assert float((globally - ALPHA).abs().max()) > 0.15
        assert float((mondrian - ALPHA).abs().max()) < 0.06


class TestDeriva:
    def test_el_adaptativo_recupera_lo_que_split_pierde(
        self, drifted: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        # Esta es la prueba que justifica el modo adaptativo. Lo que se afirma es
        # que reduce la descalibracion en un factor grande, no que la elimine:
        # bajo una rampa fuerte ningun metodo en linea llega a alpha exacto.
        calib, test = drifted
        split_rate = float(_flagged(_split().fit(calib).score(test)).mean())
        adaptive_rate = float(_flagged(_adaptive().fit(calib).score(test)).mean())

        assert split_rate > 5 * ALPHA
        assert adaptive_rate < split_rate / 3
        assert adaptive_rate < 2 * ALPHA

    def test_sin_deriva_el_adaptativo_no_estropea_la_cobertura(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        calib, test = plain
        rate = float(_flagged(_adaptive().fit(calib).score(test)).mean())
        assert rate <= ALPHA * 1.7


class TestAbsorcionDeLaAnomalia:
    def test_la_cuarentena_impide_que_el_detector_se_calle(self) -> None:
        # Sin cuarentena, ACI baja el nivel efectivo mientras la anomalia dura,
        # ensancha el intervalo y deja de marcar lo que estaba detectando. Con
        # cuarentena la sigue marcando hasta `max_freeze`, y a partir de ahi la
        # absorbe a proposito: un desplazamiento permanente acaba siendo normal.
        result = make_result(residual=homoscedastic(1.0), n_windows=200, holdout_windows=60, seed=3)
        holdout = result.windows.loc[result.windows["stage"] == "holdout"]
        start = pd.Timestamp(holdout["first_pred"].min()) + pd.Timedelta(hours=480)
        calib, test = _frames(
            inject_shift(result, uid="s00", start=start, length=240, magnitude=9.0)
        )
        span = pd.date_range(start, periods=240, freq="h")

        def flagged_head(freeze_after: int) -> float:
            scores = _adaptive(freeze_after=freeze_after).fit(calib).score(test)
            inside = scores["ds"].isin(span).to_numpy()
            return float(_flagged(scores).to_numpy()[inside][:48].mean())

        assert flagged_head(10**6) < 0.75
        assert flagged_head(3) > 0.95


class TestInformes:
    def test_el_informe_de_cobertura_publica_el_tamano_de_calibracion(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        # El numero que sostiene la garantia tiene que ser auditable, igual que
        # el denominador de MASE.
        calib, _ = plain
        report = _split().fit(calib).coverage_report()
        assert not report.empty
        assert (report["n_calib"] > 0).all()
        assert report["flag_rate"].between(0.0, 1.0).all()

    def test_el_informe_de_congelacion_cuenta_lo_excluido(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        calib, _ = plain
        report = _adaptive().fit(calib).freeze_report()
        assert set(report["unique_id"]) == {"s00"}
        assert (report["n_ingested"] > 0).all()
        assert (report["n_excluded"] <= report["n_ingested"]).all()


class TestInversionDelNivelEfectivo:
    def test_sin_aci_el_p_valor_adaptativo_es_el_estatico(self) -> None:
        # Con gamma = 0 el nivel efectivo es la propia rejilla y la inversion es
        # la identidad, tambien fuera de los extremos de la rejilla. Es lo que
        # hace que split conformal salga como caso degenerado exacto y no como
        # aproximacion.
        grid = np.asarray(ALPHA_GRID, dtype=float)
        for value in (0.0005, 0.001, 0.004, 0.05, 0.2, 0.4):
            assert _invert(value, grid, grid) == pytest.approx(min(value, 1.0), rel=1e-12)

    def test_un_nivel_efectivo_mas_alto_hace_el_score_mas_severo(self) -> None:
        grid = np.asarray(ALPHA_GRID, dtype=float)
        relaxed = _invert(0.02, grid, grid)
        strict = _invert(0.02, grid, grid * 2.0)
        assert strict < relaxed


class TestValidacion:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alpha_nominal": 0.0},
            {"alpha_ref": 0.0333},
            {"gamma": -0.1},
            {"pool_size": 0},
            {"hour_bins": 5},
            {"lead_bins": (0, 6)},
            {"lead_bins": (6, 1)},
            {"min_calib": 0},
            {"freeze_after": 0},
            {"alpha_grid": (0.2, 0.1)},
        ],
    )
    def test_una_configuracion_incoherente_falla_al_construirse(
        self, kwargs: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError):
            ConformalDetector(**kwargs)  # type: ignore[arg-type]

    def test_un_tramo_sin_cuantiles_no_se_puntua_a_ciegas(
        self, plain: tuple[ScoringFrame, ScoringFrame]
    ) -> None:
        calib, test = plain
        blind = ScoringFrame(
            df=test.df.drop(columns=["q_0250"]),
            spec=test.spec,
            model_id=test.model_id,
            start=test.start,
            end=test.end,
        )
        fitted = _split().fit(calib)
        with pytest.raises(ValueError, match="q_0250"):
            fitted.score(blind)
