"""Tests de fuga temporal pendientes de implementar.

Estan aqui, y no en una lista de tareas, porque son el criterio de aceptacion de
los hitos H2 y H4: mientras esten saltados el arnes no esta verificado, y la
suite lo dice en cada ejecucion en lugar de callarlo.

Ver docs/ARCHITECTURE.md §8.1.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="H2: features/ops.py aun no implementa las primitivas retrospectivas")
def test_t1_estabilidad_por_prefijos() -> None:
    """Para todo corte t, `features(panel[:t])` coincide con `features(panel)[:t]`.

    Es una propiedad general: detecta cualquier operacion prospectiva sin
    necesidad de enumerarlas.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="H4: el motor de backtesting aun no cierra el bucle")
def test_t2_canario_control_positivo() -> None:
    """Una columna `_canary = y` declarada `futr_exog` debe hundir el MASE.

    Si no lo hace, el canal de exogenas futuras esta desconectado y todos los
    resultados con exogenas serian basura sin que nada lo delatase.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="H4: el motor de backtesting aun no cierra el bucle")
def test_t3_canario_control_negativo() -> None:
    """La misma columna declarada `hist_exog` no debe mejorar el MASE.

    Si mejora, hay fuga por el canal historico. Junto con T2 forma el par que
    convierte "creo que no hay fuga" en "lo he medido en ambos sentidos".
    """
    raise NotImplementedError
