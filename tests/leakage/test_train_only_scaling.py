"""El escalador de la ventana k no ha visto ningun dato posterior al cutoff k.

La barrera es estructural y tiene tres piezas, ninguna de las cuales es un
comentario ni una convencion:

1. `Panel` no expone `scale`, `impute` ni `transform`, y el proyecto no tiene
   etapa global de preprocesado. No hay ningun sitio donde ajustar un escalador
   con todo el dataset, porque no existe la funcion.
2. Todo preprocesado dependiente de datos vive dentro de `Forecaster.fit`, que
   solo recibe ``panel.train(window)``.
3. El motor comprueba antes de entregarlo que ese panel no cruza el cutoff.

Lo que estos tests miden es la consecuencia observable: los estadisticos que el
modelo calculo coinciden **exactamente** con los que se obtienen del tramo de
entrenamiento de su ventana, y no con los del panel completo. Para que la
diferencia sea imposible de confundir con ruido numerico, el panel esta
envenenado: a partir del primer cutoff la serie se multiplica por mil. Un
escalador que hubiese visto un solo punto posterior lo delataria por ordenes de
magnitud.
"""

from __future__ import annotations

import numpy as np
import pytest

from chronolab.data.futr import RealizedFutrProvider
from chronolab.errors import PerfectForesightWarning
from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.evaluation.splitters import Window
from chronolab.panel import Panel
from tests.fixtures.models import ScaledExogProbe
from tests.fixtures.synthetic import WEEKLY, hourly_spec, make_hourly_frame

PLAN = BacktestPlan(h=24, n_windows=3, step_size=48)
POISON = 1_000.0


def _scaler(values: np.ndarray) -> tuple[float, float]:
    """Media y desviacion tipica poblacional, ignorando huecos.

    Es la misma definicion que usa el modelo sonda. Se reimplementa aqui a
    proposito: si el test llamase a la funcion del modelo, ambos podrian estar
    equivocados de la misma manera.
    """
    observed = values[~np.isnan(values)]
    std = float(observed.std())
    return float(observed.mean()), std if std > 0 else 1.0


def _poisoned_panel() -> tuple[Panel, tuple[Window, ...]]:
    """Panel cuya serie se dispara justo despues del primer cutoff del plan."""
    frame = make_hourly_frame(n_series=2, n_hours=WEEKLY * 8, seed=11)
    panel = Panel(df=frame, spec=hourly_spec())
    windows = PLAN.splitter().split(panel)

    poisoned = frame.copy()
    after_first_cutoff = poisoned["ds"] > windows[0].cutoff
    poisoned.loc[after_first_cutoff, "y"] = poisoned.loc[after_first_cutoff, "y"] * POISON
    return Panel(df=poisoned, spec=hourly_spec()), windows


@pytest.fixture(scope="module")
def sonda() -> tuple[ScaledExogProbe, Panel, tuple[Window, ...]]:
    panel, windows = _poisoned_panel()
    probe = ScaledExogProbe()
    with pytest.warns(PerfectForesightWarning):
        provider = RealizedFutrProvider(panel=panel)
    backtest(panel, [probe], PLAN, futr=provider)
    return probe, panel, windows


def test_el_escalador_de_cada_ventana_coincide_con_su_tramo_de_entrenamiento(
    sonda: tuple[ScaledExogProbe, Panel, tuple[Window, ...]],
) -> None:
    probe, panel, windows = sonda
    assert len(probe.fits) == len(windows)

    for record, window in zip(probe.fits, windows, strict=True):
        train = panel.train(window)
        for uid, group in train.df.groupby("unique_id", sort=False):
            expected = _scaler(group["y"].to_numpy(dtype=float))
            assert record.scaler[str(uid)] == pytest.approx(expected, rel=1e-6)


def test_el_escalador_de_la_primera_ventana_no_ha_visto_el_veneno(
    sonda: tuple[ScaledExogProbe, Panel, tuple[Window, ...]],
) -> None:
    probe, panel, _ = sonda

    for uid, group in panel.df.groupby("unique_id", sort=False):
        global_mean, _ = _scaler(group["y"].to_numpy(dtype=float))
        window_mean, _ = probe.fits[0].scaler[str(uid)]
        # El panel completo esta dominado por el tramo envenenado; el tramo de
        # entrenamiento de la primera ventana no lo toca en absoluto. La
        # separacion es de casi dos ordenes de magnitud: no hay forma de
        # confundirla con ruido numerico ni con una diferencia de convenio.
        assert window_mean * 10 < global_mean


def test_el_escalador_se_reajusta_en_cada_ventana(
    sonda: tuple[ScaledExogProbe, Panel, tuple[Window, ...]],
) -> None:
    # Un escalador ajustado una sola vez para todo el run seria indistinguible
    # de uno correcto si solo se mirase la primera ventana.
    probe, _, _ = sonda
    medias = [record.scaler["s00"][0] for record in probe.fits]

    assert len(set(medias)) == len(medias)
    assert medias == sorted(medias)  # el veneno entra en el entrenamiento poco a poco


def test_ningun_ajuste_recibio_datos_posteriores_a_su_cutoff(
    sonda: tuple[ScaledExogProbe, Panel, tuple[Window, ...]],
) -> None:
    probe, panel, windows = sonda

    for record, window in zip(probe.fits, windows, strict=True):
        assert record.train_last_ds == window.cutoff
        assert record.train_first_ds == window.train_start
        # Y el volumen cuadra con la rejilla: ni una fila de mas.
        expected_rows = len(panel.grid()[: panel.grid().get_loc(window.cutoff) + 1]) * len(
            panel.ids()
        )
        assert record.n_rows == expected_rows


def test_las_predicciones_no_reciben_la_columna_objetivo(
    sonda: tuple[ScaledExogProbe, Panel, tuple[Window, ...]],
) -> None:
    probe, _, windows = sonda

    for record, window in zip(probe.predictions, windows, strict=True):
        assert "y" not in record.futr_columns
        assert record.futr_first_ds == window.first_pred
        assert record.futr_last_ds == window.last_pred
