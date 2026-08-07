"""Fuga L9: la calibracion de un detector no puede ver lo que va a puntuar.

Hay dos formas de que un detector conformal mire hacia delante, y aqui se
comprueban las dos. La primera es obvia: calibrar sobre el tramo que se puntua.
La segunda no lo es: realimentar el estado en linea con una observacion que en el
origen de la prediccion todavia no existia. Un detector con ese fallo mejora sus
numeros sin que ninguna asercion de cutoff salte, porque nada de lo que usa es
posterior al **instante puntuado** —solo al instante en que se emitio la banda—.
"""

from __future__ import annotations

import pandas as pd
import pytest

from chronolab.anomaly.conformal import ConformalDetector
from chronolab.artifacts.reader import scoring_frame
from chronolab.errors import CutoffViolation
from chronolab.evaluation.backtest import BacktestResult
from tests.fixtures.anomaly import MODEL, homoscedastic, make_result


def _detector() -> ConformalDetector:
    return ConformalDetector(
        base_model_id=MODEL,
        hour_bins=1,
        min_calib=200,
        gamma=0.02,
        pool_size=250,
    )


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    return make_result(residual=homoscedastic(1.0), n_windows=120, holdout_windows=30, seed=5)


def test_la_calibracion_no_se_solapa_con_lo_que_se_puntua(result: BacktestResult) -> None:
    calib = scoring_frame(result, model_id=MODEL, stage="dev")
    test = scoring_frame(result, model_id=MODEL, stage="holdout")
    fitted = _detector().fit(calib)

    assert fitted.cutoff == calib.end
    assert (test.df["ds"] > fitted.cutoff).all()
    with pytest.raises(CutoffViolation):
        fitted.score(calib)


def test_el_estado_no_se_realimenta_con_lo_que_aun_no_se_conocia(
    result: BacktestResult,
) -> None:
    # Canario: se altera la observacion del ultimo instante de una ventana del
    # tramo puntuado. Esa observacion solo existe *despues* del origen de esa
    # ventana, asi que no ha podido entrar en ninguna banda de la ventana. Si
    # algun otro punto de la misma ventana cambia de score, el detector se esta
    # realimentando con informacion que en el momento de predecir no tenia.
    calib = scoring_frame(result, model_id=MODEL, stage="holdout")
    windows = result.windows.loc[result.windows["stage"] == "holdout"]
    target = windows.iloc[len(windows) // 2]
    last_ds = pd.Timestamp(target["last_pred"])

    fitted = _detector().fit(scoring_frame(result, model_id=MODEL, stage="dev"))
    clean = fitted.score(calib)

    tampered = result.forecasts.copy()
    canary = (tampered["ds"] == last_ds) & (tampered["unique_id"] == "s00")
    assert canary.any()
    tampered.loc[canary, "y"] = tampered.loc[canary, "y"] + 500.0

    poisoned = fitted.score(
        scoring_frame(
            BacktestResult(
                forecasts=tampered,
                windows=result.windows,
                model_runs=result.model_runs,
                spec=result.spec,
                plan=result.plan,
                futr_vintage=None,
            ),
            model_id=MODEL,
            stage="holdout",
        )
    )

    # Todo lo anterior o igual al final de esa ventana, salvo el propio canario,
    # tiene que salir identico.
    untouched = (clean["ds"] <= last_ds) & ~(
        (clean["ds"] == last_ds) & (clean["unique_id"] == "s00")
    )
    pd.testing.assert_frame_equal(
        clean.loc[untouched].reset_index(drop=True),
        poisoned.loc[untouched].reset_index(drop=True),
    )


def test_el_canario_si_afecta_a_lo_que_viene_despues(result: BacktestResult) -> None:
    # La otra mitad del canario: si alterar la observacion no cambiara **nada**,
    # el test anterior pasaria con un detector que ignora la realimentacion.
    calib = scoring_frame(result, model_id=MODEL, stage="dev")
    test = scoring_frame(result, model_id=MODEL, stage="holdout")
    windows = result.windows.loc[result.windows["stage"] == "holdout"]
    target = windows.iloc[len(windows) // 2]
    last_ds = pd.Timestamp(target["last_pred"])

    fitted = _detector().fit(calib)
    clean = fitted.score(test)

    tampered = result.forecasts.copy()
    canary = (tampered["ds"] == last_ds) & (tampered["unique_id"] == "s00")
    tampered.loc[canary, "y"] = tampered.loc[canary, "y"] + 500.0
    poisoned = fitted.score(
        scoring_frame(
            BacktestResult(
                forecasts=tampered,
                windows=result.windows,
                model_runs=result.model_runs,
                spec=result.spec,
                plan=result.plan,
                futr_vintage=None,
            ),
            model_id=MODEL,
            stage="holdout",
        )
    )

    after = clean["ds"] > last_ds
    assert not clean.loc[after, "score"].equals(poisoned.loc[after, "score"])
