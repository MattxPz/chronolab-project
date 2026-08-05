"""Canario: una exogena que es `y` desplazada hacia el futuro, medida en los dos sentidos.

Es el par de controles de docs/ARCHITECTURE.md §8.1 (T2 y T3) aplicado a la
columna mas peligrosa que se puede inyectar en un panel: `_canary[t] = y[t + 1]`,
es decir el valor que la serie tendra en el instante siguiente. Un modelo que la
tenga disponible en el tramo de prediccion acierta casi perfectamente.

La pregunta que responden estos tests no es "¿hay fuga?" sino las dos que
importan, en orden:

1. **¿Funciona el canal de exogenas futuras?** Si el canario declarado
   `futr_exog` **no** hunde el error, es que el canal esta desconectado, y
   entonces cualquier resultado posterior sobre exogenas es basura sin sintoma:
   los modelos con exogenas y sin ellas darian lo mismo y nadie sabria por que.
   Sin este control positivo, el test negativo no prueba nada, porque un modelo
   que ignora sus entradas tambien pasa.
2. **¿Esta cortado el canal historico?** El mismo canario declarado `hist_exog`
   no debe mejorar nada, porque en el tramo posterior al cutoff ese valor no
   existe para el modelo.

Y una tercera, previa a ejecutar nada: construir la columna con `lead()` sobre la
objetivo ni siquiera se puede escribir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronolab.data.futr import RealizedFutrProvider
from chronolab.errors import PerfectForesightWarning
from chronolab.evaluation.backtest import BacktestPlan, BacktestResult, backtest
from chronolab.features import ops
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId
from tests.fixtures.models import ScaledExogProbe
from tests.fixtures.synthetic import DAILY, WEEKLY, make_hourly_frame

PLAN = BacktestPlan(h=24, n_windows=4, step_size=24)
CANARY = "_canary"


def _panel_con_canario(role: str | None) -> Panel:
    """Panel sintetico con `_canary[t] = y[t + 1]` declarada en el rol indicado.

    Con ``role=None`` la columna se calcula igualmente y luego se descarta: los
    tres paneles cubren exactamente el mismo tramo temporal, de modo que las tres
    comparaciones se hacen sobre las mismas ventanas y los mismos instantes. Si
    la linea base tuviese una hora mas, la diferencia de error incluiria esa hora
    y ya no mediria el canal de exogenas.
    """
    frame = make_hourly_frame(n_series=2, n_hours=WEEKLY * 6, seed=7)
    columns = {"futr_exog": ("temp_c",), "hist_exog": ("voltage",)}

    # y desplazada hacia el futuro: el valor de la hora siguiente.
    frame[CANARY] = frame.groupby("unique_id", sort=False)["y"].shift(-1)
    # La ultima fila de cada serie no tiene futuro; se recorta el panel para no
    # introducir huecos que confundan el efecto con un problema de NaN.
    frame = frame[frame[CANARY].notna()].reset_index(drop=True)
    if role is not None:
        columns[role] = (*columns[role], CANARY)

    spec = PanelSpec(
        dataset_id=DatasetId("canary_h"),
        freq="h",
        seasonalities=(DAILY, WEEKLY),
        futr_exog=columns["futr_exog"],
        hist_exog=columns["hist_exog"],
    )
    return Panel(df=frame[list(spec.columns)], spec=spec)


def _mae(result: BacktestResult) -> float:
    """Error absoluto medio del run, sobre todas las series y ventanas."""
    errors = (result.forecasts["y"] - result.forecasts["y_hat"]).abs()
    return float(errors.mean())


def _run(panel: Panel) -> BacktestResult:
    """Backtest con el modelo sonda, que usa exactamente lo que le llega en el `FutrFrame`."""
    with pytest.warns(PerfectForesightWarning):
        provider = RealizedFutrProvider(panel=panel)
    return backtest(panel, [ScaledExogProbe()], PLAN, futr=provider)


@pytest.fixture(scope="module")
def baseline() -> BacktestResult:
    return _run(_panel_con_canario(None))


@pytest.fixture(scope="module")
def como_futura() -> BacktestResult:
    return _run(_panel_con_canario("futr_exog"))


@pytest.fixture(scope="module")
def como_historica() -> BacktestResult:
    return _run(_panel_con_canario("hist_exog"))


def test_construir_el_canario_con_lead_sobre_la_objetivo_es_imposible() -> None:
    """`lead(y, k)` falla al construirse: la fuga se rechaza antes de existir.

    Esta es la version fuerte de "el motor la rechaza": no hace falta llegar a
    ejecutar un backtest, porque la columna no se puede fabricar con las
    primitivas del proyecto.
    """
    panel = _panel_con_canario(None)

    with pytest.raises(ValueError, match="fuga por construccion"):
        ops.lead(panel, "y", 1)
    with pytest.raises(ValueError, match="fuga por construccion"):
        ops.lead(panel, "voltage", 24)

    # Sobre una exogena conocida a futuro si es legitimo: el calendario y una
    # prevision existen en el cutoff para instantes posteriores.
    assert np.isinf(ops.lead(panel, "temp_c", 3).max_lead)


class TestT2ControlPositivo:
    """El canal de exogenas futuras funciona: si se conecta la fuga, se ve."""

    def test_el_canario_como_exogena_futura_hunde_el_error(
        self, baseline: BacktestResult, como_futura: BacktestResult
    ) -> None:
        # No es un resultado deseable: es el control que demuestra que el test
        # negativo tiene poder de deteccion.
        assert _mae(como_futura) < 0.25 * _mae(baseline)

    def test_el_modelo_recibe_el_canario_en_la_trama_de_futuras(
        self, como_futura: BacktestResult
    ) -> None:
        del como_futura  # el efecto se comprueba sobre la sonda, no sobre el artefacto
        panel = _panel_con_canario("futr_exog")
        probe = ScaledExogProbe()
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        backtest(panel, [probe], PLAN, futr=provider)

        assert all(CANARY in record.futr_columns for record in probe.predictions)


class TestT3ControlNegativo:
    """El canal historico esta cortado: la misma columna, declarada `hist_exog`, no llega."""

    def test_el_canario_como_exogena_historica_no_mejora_el_error(
        self, baseline: BacktestResult, como_historica: BacktestResult
    ) -> None:
        # Identico, no "parecido": la columna no llega al tramo de prediccion, de
        # modo que el modelo predice exactamente lo mismo que sin ella.
        assert _mae(como_historica) == pytest.approx(_mae(baseline), rel=1e-9)

    def test_el_modelo_no_ve_el_canario_en_ninguna_ventana(self) -> None:
        panel = _panel_con_canario("hist_exog")
        probe = ScaledExogProbe()
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        backtest(panel, [probe], PLAN, futr=provider)

        assert probe.predictions, "la sonda no ha predicho en ninguna ventana"
        for record in probe.predictions:
            # Ausencia fisica: no esta omitida por convenio, no existe en la trama.
            assert CANARY not in record.futr_columns
            assert "voltage" not in record.futr_columns
            assert "y" not in record.futr_columns

    def test_el_canario_historico_si_esta_en_el_entrenamiento(self) -> None:
        # El corte es en el cutoff, no en la columna: en el pasado la exogena
        # historica es un dato legitimo y el modelo puede usarla.
        panel = _panel_con_canario("hist_exog")
        probe = ScaledExogProbe()
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        backtest(panel, [probe], PLAN, futr=provider)

        assert all(CANARY in record.train_columns for record in probe.fits)


def test_el_error_del_canario_futuro_es_casi_cero(como_futura: BacktestResult) -> None:
    """Con el valor de la hora siguiente disponible, el error cae al nivel del ruido.

    Es la comprobacion que da sentido a la escala: sin un numero absoluto, un
    "mejora mucho" podria seguir siendo una mejora modesta.
    """
    escala = float(pd.Series(como_futura.forecasts["y"]).std())
    assert _mae(como_futura) < 0.2 * escala
