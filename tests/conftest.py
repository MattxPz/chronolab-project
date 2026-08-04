"""Fixtures compartidas de la suite.

Las series sinteticas son la base del arnes: tienen estacionalidad conocida
(diaria 24 y semanal 168), rejilla completa y `ds` en UTC ingenuo, que es
exactamente el contrato del panel canonico.
"""

from __future__ import annotations

import pandas as pd
import pytest

from chronolab.panel import Panel, PanelSpec
from tests.fixtures.synthetic import hourly_spec, make_hourly_frame, make_hourly_panel


@pytest.fixture
def spec() -> PanelSpec:
    """Especificacion del panel horario sintetico."""
    return hourly_spec()


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    """Trama larga horaria con tres series y doce semanas."""
    return make_hourly_frame()


@pytest.fixture
def hourly_panel() -> Panel:
    """Panel horario sintetico con tres series y doce semanas."""
    return make_hourly_panel()


@pytest.fixture
def single_series_frame() -> pd.DataFrame:
    """Trama larga horaria de una sola serie, util para tests de una sola dimension."""
    return make_hourly_frame(n_series=1)
