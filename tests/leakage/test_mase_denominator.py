"""El denominador de MASE no ha visto nada posterior al cutoff de su ventana.

MASE escala el error del tramo evaluado por el error que cometeria un naive
estacional **sobre el entrenamiento**. Calcularlo sobre el test, o sobre la serie
completa, o una sola vez para todo el run, es la fuga mas comun del dominio: no
rompe nada, no cambia el signo de las comparaciones de forma evidente, y produce
un numero que sigue pareciendo un MASE.

La barrera es que `mase_denominators` solo puede llegar al panel a traves de
``panel.slice(train_start, cutoff)``, que es exactamente el mismo tramo que
recibio `Forecaster.fit`. Estos tests miden la consecuencia observable: envenenar
todo lo que hay despues del cutoff de una ventana no mueve su denominador ni una
diezmilesima, y en cambio si mueve —por ordenes de magnitud— al que se calculase
mal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronolab.evaluation.aggregate import score_forecasts
from chronolab.evaluation.backtest import BacktestPlan, BacktestResult, backtest
from chronolab.evaluation.metrics import mase, mase_denominators, seasonal_naive_mae
from chronolab.panel import Panel
from tests.fixtures.models import SeasonalNaiveProbe
from tests.fixtures.synthetic import DAILY, WEEKLY, hourly_spec, make_hourly_frame

PLAN = BacktestPlan(h=24, n_windows=3, step_size=48)
POISON = 1_000.0


@pytest.fixture(scope="module")
def panel() -> Panel:
    return Panel(df=make_hourly_frame(n_series=2, n_hours=WEEKLY * 6, seed=17), spec=hourly_spec())


@pytest.fixture(scope="module")
def result(panel: Panel) -> BacktestResult:
    return backtest(panel, [SeasonalNaiveProbe(season=DAILY)], PLAN)


def _poisoned_after(panel: Panel, cutoff: pd.Timestamp) -> Panel:
    """Panel identico hasta `cutoff` y multiplicado por mil a partir de ahi."""
    frame = panel.df.copy()
    frame.loc[frame["ds"] > cutoff, "y"] = frame.loc[frame["ds"] > cutoff, "y"] * POISON
    return Panel(df=frame, spec=panel.spec)


def test_envenenar_el_futuro_no_mueve_el_denominador_de_ninguna_ventana(
    panel: Panel, result: BacktestResult
) -> None:
    limpio = mase_denominators(panel, result.windows)

    for window in result.windows.itertuples(index=False):
        una_ventana = result.windows[result.windows["window_id"] == window.window_id]
        envenenado = mase_denominators(_poisoned_after(panel, window.cutoff), una_ventana)
        esperado = limpio[limpio["window_id"] == window.window_id]

        pd.testing.assert_frame_equal(
            envenenado.reset_index(drop=True), esperado.reset_index(drop=True)
        )


def test_calcularlo_sobre_la_serie_completa_daria_otro_numero(
    panel: Panel, result: BacktestResult
) -> None:
    # Control positivo: si la barrera no existiese, el efecto seria enorme. Sin
    # esta comprobacion, el test anterior pasaria igual con una serie en la que
    # el veneno no cambiase nada.
    ventana = result.windows.iloc[0]
    envenenado = _poisoned_after(panel, pd.Timestamp(ventana["cutoff"]))
    serie = envenenado.df[envenenado.df["unique_id"] == "s00"]["y"].to_numpy(dtype=float)

    correcto = mase_denominators(envenenado, result.windows.head(1))
    correcto_s00 = correcto.loc[correcto["unique_id"] == "s00", "mase_denominator"].item()
    global_ = seasonal_naive_mae(serie, season=DAILY)

    assert global_ > 100 * correcto_s00


def test_el_denominador_cambia_de_una_ventana_a_otra(result: BacktestResult, panel: Panel) -> None:
    # Un unico denominador para todo el run seria indistinguible de uno correcto
    # mirando solo una ventana. Con origen rodante y entrenamiento expansivo,
    # cada ventana tiene su propia escala.
    denominadores = mase_denominators(panel, result.windows)
    por_serie = denominadores.groupby("unique_id")["mase_denominator"].nunique()
    assert (por_serie == len(result.windows)).all()


def test_el_mase_del_run_se_reproduce_solo_con_datos_de_entrenamiento(
    panel: Panel, result: BacktestResult
) -> None:
    # Reconstruccion independiente: se recalcula el denominador recortando el
    # panel a mano, sin pasar por `mase_denominators`, y el MASE resultante
    # coincide. Es lo que convierte "no hay fuga" en algo que un tercero puede
    # verificar con el panel y la tabla de ventanas delante.
    scored = score_forecasts(result, panel)

    a_mano: list[float] = []
    for row in scored.itertuples(index=False):
        train = panel.slice(
            pd.Timestamp(result.windows.loc[row.window_id, "train_start"]),
            pd.Timestamp(result.windows.loc[row.window_id, "cutoff"]),
        )
        serie = train.df[train.df["unique_id"] == row.unique_id]["y"].to_numpy(dtype=float)
        a_mano.append(seasonal_naive_mae(serie, season=DAILY))

    np.testing.assert_allclose(scored["mase_denominator"].to_numpy(), a_mano, rtol=1e-9)
    assert mase(
        scored["y"].to_numpy(), scored["y_hat"].to_numpy(), denominator=np.array(a_mano)
    ) == pytest.approx(
        mase(
            scored["y"].to_numpy(),
            scored["y_hat"].to_numpy(),
            denominator=scored["mase_denominator"].to_numpy(),
        )
    )
