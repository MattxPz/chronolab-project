"""`scoring_frame`: unico constructor de `ScoringFrame`.

Lo que se comprueba aqui es que el corte entre calibracion y puntuacion cae donde
el splitter ya lo puso —la frontera dev/holdout— y que un plan que no lo permita
se detiene en lugar de producir un tramo ambiguo.
"""

from __future__ import annotations

import pandas as pd
import pytest

from chronolab.artifacts.reader import SCORING_FRAME_COLUMNS, scoring_frame
from chronolab.errors import ArtifactNotFound, WindowValidationError
from chronolab.evaluation.backtest import BacktestPlan, BacktestResult
from chronolab.types import ModelId
from tests.fixtures.anomaly import MODEL, drop_window, homoscedastic, make_result


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    return make_result(residual=homoscedastic(1.0), n_series=2, n_windows=20, holdout_windows=6)


class TestEsquema:
    def test_trae_el_cutoff_y_el_paso_del_horizonte(self, result: BacktestResult) -> None:
        # Sin ellos se pierden el adelanto —la escala del residuo— y la frontera
        # de informacion de la banda, que es lo que un detector en linea necesita.
        frame = scoring_frame(result, model_id=MODEL, stage="holdout")
        for column in SCORING_FRAME_COLUMNS:
            assert column in frame.df.columns
        assert any(column.startswith("q_") for column in frame.df.columns)

    def test_el_adelanto_es_coherente_con_el_cutoff(self, result: BacktestResult) -> None:
        frame = scoring_frame(result, model_id=MODEL, stage="holdout")
        df = frame.df
        lead = (df["ds"] - df["cutoff"]) // pd.Timedelta(1, "h")
        assert (lead == df["h_step"] + result.plan.gap).all()

    def test_todo_lo_puntuable_es_posterior_a_su_origen(self, result: BacktestResult) -> None:
        df = scoring_frame(result, model_id=MODEL, stage="holdout").df
        assert (df["ds"] > df["cutoff"]).all()

    def test_una_fila_por_serie_e_instante_sobre_la_rejilla_completa(
        self, result: BacktestResult
    ) -> None:
        frame = scoring_frame(result, model_id=MODEL, stage="holdout")
        grid = pd.date_range(frame.start, frame.end, freq=result.spec.freq)
        assert len(frame.df) == len(grid) * 2
        assert not frame.df.duplicated(subset=["unique_id", "ds"]).any()
        assert frame.df.equals(frame.df.sort_values(["unique_id", "ds"]).reset_index(drop=True))


class TestCorteEntreEtapas:
    def test_la_calibracion_termina_antes_de_lo_que_se_puntua(self, result: BacktestResult) -> None:
        calib = scoring_frame(result, model_id=MODEL, stage="dev")
        test = scoring_frame(result, model_id=MODEL, stage="holdout")
        assert calib.end < test.start

    def test_los_dos_tramos_son_contiguos_y_disjuntos(self, result: BacktestResult) -> None:
        # El plan es teselado, asi que entre el ultimo instante calibrado y el
        # primero puntuado hay exactamente un paso: ni solape ni agujero.
        calib = scoring_frame(result, model_id=MODEL, stage="dev")
        test = scoring_frame(result, model_id=MODEL, stage="holdout")
        assert test.start - calib.end == pd.Timedelta(1, "h")
        assert set(calib.df["ds"]).isdisjoint(set(test.df["ds"]))


class TestPlanesQueNoAdmitenDeteccion:
    def test_un_plan_con_solape_se_detiene(self, result: BacktestResult) -> None:
        # Con step_size < h un instante tendria varias predicciones y podria caer
        # en dev y en holdout a la vez.
        overlapping = BacktestPlan(h=24, n_windows=20, step_size=12, holdout_windows=6)
        broken = BacktestResult(
            forecasts=result.forecasts,
            windows=result.windows,
            model_runs=result.model_runs,
            spec=result.spec,
            plan=overlapping,
            futr_vintage=None,
        )
        with pytest.raises(WindowValidationError, match="teselado"):
            scoring_frame(broken, model_id=MODEL, stage="holdout")

    def test_un_plan_sin_holdout_se_detiene(self, result: BacktestResult) -> None:
        without = BacktestPlan(h=24, n_windows=20, step_size=24, holdout_windows=0)
        broken = BacktestResult(
            forecasts=result.forecasts,
            windows=result.windows,
            model_runs=result.model_runs,
            spec=result.spec,
            plan=without,
            futr_vintage=None,
        )
        with pytest.raises(WindowValidationError, match="holdout"):
            scoring_frame(broken, model_id=MODEL, stage="holdout")

    def test_un_modelo_que_no_esta_en_el_run(self, result: BacktestResult) -> None:
        with pytest.raises(ArtifactNotFound, match="ausente"):
            scoring_frame(result, model_id=ModelId("ausente"), stage="holdout")


class TestHuecos:
    def test_una_ventana_fallida_deja_nulos_y_no_filas_ausentes(
        self, result: BacktestResult
    ) -> None:
        # Un hueco tiene que poder distinguirse de un instante no evaluado: si
        # desapareciese, la rejilla dejaria de estar alineada entre detectores.
        holdout = result.windows.loc[result.windows["stage"] == "holdout", "window_id"]
        broken_id = int(holdout.iloc[2])
        complete = scoring_frame(result, model_id=MODEL, stage="holdout")
        holed = scoring_frame(
            drop_window(result, window_id=broken_id, uid="s00"),
            model_id=MODEL,
            stage="holdout",
        )

        assert len(holed.df) == len(complete.df)
        missing = holed.df["y"].isna()
        assert missing.sum() == result.plan.h
        assert holed.df.loc[missing, "cutoff"].isna().all()
        assert holed.df.loc[missing, "h_step"].isna().all()
        assert (holed.df.loc[missing, "unique_id"] == "s00").all()
