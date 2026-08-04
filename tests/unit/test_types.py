"""Tipos base: roles, vintages y version del paquete."""

from __future__ import annotations

import chronolab
from chronolab.types import Role, Vintage


class TestRole:
    def test_distingue_exogenas_futuras_de_historicas(self) -> None:
        assert Role.FUTR_EXOG != Role.HIST_EXOG

    def test_los_valores_son_estables(self) -> None:
        # Se serializan en `conf/datasets.yaml` y en los artefactos: cambiarlos
        # invalidaria runs anteriores.
        assert Role.TARGET == "target"
        assert Role.FUTR_EXOG == "futr_exog"
        assert Role.HIST_EXOG == "hist_exog"
        assert Role.STATIC_EXOG == "static_exog"


class TestVintage:
    def test_los_tres_vintages_son_distintos(self) -> None:
        assert len(set(Vintage)) == 3

    def test_los_valores_son_estables(self) -> None:
        # Entran en el hash de configuracion y se persisten en la tabla `runs`.
        assert Vintage.REALIZED == "realized"
        assert Vintage.ARCHIVED_FORECAST == "archived_forecast"
        assert Vintage.SIMULATED_FORECAST == "simulated_forecast"


class TestPaquete:
    def test_expone_version(self) -> None:
        assert chronolab.__version__

    def test_expone_el_contrato_de_datos(self) -> None:
        for name in ("Panel", "PanelSpec", "FutrFrame", "Role", "Vintage"):
            assert name in chronolab.__all__
            assert hasattr(chronolab, name)
