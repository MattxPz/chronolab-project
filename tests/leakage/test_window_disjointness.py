"""Ninguna ventana de evaluacion se solapa con su propio entrenamiento.

Es la propiedad mas basica del origen rodante y la que mas veces se rompe por un
off-by-one en el `gap` o por un intervalo cerrado donde deberia ser semiabierto.
Se comprueba de tres maneras, porque cada una detecta un fallo distinto:

- **Sobre la aritmetica de la ventana**, en toda una malla de configuraciones:
  detecta un splitter que calcula mal los extremos.
- **Sobre los datos que realmente se entregan**, cruzando el `Panel` que recibe
  `fit` con las filas que se evaluan: detecta un splitter correcto cuyas
  rebanadas no lo son. Una ventana puede ser aritmeticamente impecable y aun asi
  entregar una fila de mas si el corte es inclusivo por un lado de mas.
- **Sobre paneles generados con hypothesis**: detecta los casos limite que una
  malla elegida a mano no toca, empezando por ``step_size < h``, donde los
  tramos de evaluacion de ventanas *distintas* si se solapan y es facil concluir
  por error que el solape esta permitido tambien dentro de una ventana.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chronolab.evaluation.splitters import RollingOriginSplitter
from chronolab.panel import Panel
from tests.fixtures.synthetic import WEEKLY, hourly_spec, make_hourly_frame

CONFIGURACIONES = [
    pytest.param({"h": 24, "n_windows": 5, "step_size": 24}, id="expansiva-sin-gap"),
    pytest.param({"h": 24, "n_windows": 5, "step_size": 24, "gap": 12}, id="expansiva-con-gap"),
    pytest.param({"h": 48, "n_windows": 6, "step_size": 12}, id="evaluaciones-solapadas"),
    pytest.param(
        {"h": 24, "n_windows": 4, "step_size": 24, "mode": "sliding", "train_size": 336},
        id="deslizante",
    ),
    pytest.param(
        {
            "h": 12,
            "n_windows": 4,
            "step_size": 36,
            "gap": 6,
            "mode": "sliding",
            "train_size": 500,
        },
        id="deslizante-con-gap",
    ),
    pytest.param({"h": 1, "n_windows": 3, "step_size": 1}, id="horizonte-minimo"),
]


@pytest.fixture(scope="module")
def panel() -> Panel:
    return Panel(df=make_hourly_frame(n_series=2, n_hours=WEEKLY * 6, seed=3), spec=hourly_spec())


@pytest.mark.parametrize("config", CONFIGURACIONES)
def test_la_aritmetica_de_la_ventana_no_permite_solape(
    panel: Panel, config: dict[str, object]
) -> None:
    windows = RollingOriginSplitter(**config).split(panel)  # type: ignore[arg-type]

    assert windows
    for window in windows:
        assert window.train_start <= window.cutoff
        assert window.cutoff < window.first_pred
        assert window.first_pred <= window.last_pred
        # El gap declarado es el gap real, ni uno mas ni uno menos.
        assert window.first_pred - window.cutoff == pd.Timedelta(hours=window.gap + 1)


@pytest.mark.parametrize("config", CONFIGURACIONES)
def test_las_filas_entregadas_a_fit_no_estan_entre_las_evaluadas(
    panel: Panel, config: dict[str, object]
) -> None:
    windows = RollingOriginSplitter(**config).split(panel)  # type: ignore[arg-type]

    for window in windows:
        train = panel.train(window)
        evaluated = panel.actuals(window)

        entrenadas = set(zip(train.df["unique_id"], train.df["ds"], strict=True))
        evaluadas = set(zip(evaluated["unique_id"], evaluated["ds"], strict=True))

        assert entrenadas, f"ventana {window.window_id} sin entrenamiento"
        assert evaluadas, f"ventana {window.window_id} sin evaluacion"
        assert not entrenadas & evaluadas
        # Y no solo son disjuntos: el gap declarado los separa de verdad.
        assert train.last_ds == window.cutoff
        assert evaluated["ds"].min() - train.last_ds == pd.Timedelta(hours=window.gap + 1)


@pytest.mark.parametrize("config", CONFIGURACIONES)
def test_el_tramo_evaluado_dura_exactamente_h_pasos(
    panel: Panel, config: dict[str, object]
) -> None:
    windows = RollingOriginSplitter(**config).split(panel)  # type: ignore[arg-type]

    for window in windows:
        evaluated = panel.actuals(window)
        assert evaluated["ds"].nunique() == window.h
        assert len(evaluated) == window.h * len(panel.ids())


@settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    h=st.integers(min_value=1, max_value=72),
    gap=st.integers(min_value=0, max_value=48),
    step_size=st.integers(min_value=1, max_value=72),
    n_windows=st.integers(min_value=1, max_value=6),
    train_size=st.one_of(st.none(), st.integers(min_value=48, max_value=400)),
)
def test_ninguna_configuracion_produce_solape(
    panel: Panel,
    h: int,
    gap: int,
    step_size: int,
    n_windows: int,
    train_size: int | None,
) -> None:
    splitter = RollingOriginSplitter(
        h=h,
        n_windows=n_windows,
        step_size=step_size,
        gap=gap,
        mode="expanding" if train_size is None else "sliding",
        train_size=train_size,
    )
    windows = splitter.split(panel)

    for window in windows:
        train = panel.train(window)
        evaluated = panel.actuals(window)
        assert train.last_ds < evaluated["ds"].min()
        assert train.last_ds == window.cutoff
        assert evaluated["ds"].max() == window.last_pred
