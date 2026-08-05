"""`RealizedFutrProvider`: ausencia fisica de las historicas y vintage declarado."""

from __future__ import annotations

import pandas as pd
import pytest

from chronolab.data.futr import FutrProvider, RealizedFutrProvider
from chronolab.errors import PanelValidationError, PerfectForesightWarning
from chronolab.evaluation.splitters import RollingOriginSplitter, Window
from chronolab.panel import Panel
from chronolab.types import Vintage


@pytest.fixture
def window(hourly_panel: Panel) -> Window:
    return RollingOriginSplitter(h=24, n_windows=2, step_size=24).split(hourly_panel)[0]


@pytest.fixture
def provider(hourly_panel: Panel) -> RealizedFutrProvider:
    with pytest.warns(PerfectForesightWarning):
        return RealizedFutrProvider(panel=hourly_panel)


class TestVintage:
    def test_avisa_de_la_presciencia_al_construirse(self, hourly_panel: Panel) -> None:
        # El aviso va en la construccion, no en el uso: quien lo instancia ya ha
        # tomado la decision, y repetirlo por ventana solo generaria ruido.
        with pytest.warns(PerfectForesightWarning, match="cota superior"):
            RealizedFutrProvider(panel=hourly_panel)

    def test_declara_su_vintage(self, provider: RealizedFutrProvider) -> None:
        assert provider.vintage is Vintage.REALIZED

    def test_cumple_el_protocolo(self, provider: RealizedFutrProvider) -> None:
        assert isinstance(provider, FutrProvider)


class TestFutrFrame:
    def test_contiene_solo_las_exogenas_futuras(
        self, provider: RealizedFutrProvider, window: Window, hourly_panel: Panel
    ) -> None:
        futr = provider.futr(window, ids=hourly_panel.ids())

        # `voltage` es historica y `y` es la objetivo: no estan omitidas por
        # convenio, no existen en la estructura que recibe el modelo.
        assert list(futr.df.columns) == ["unique_id", "ds", "temp_c"]

    def test_cubre_exactamente_el_tramo_evaluado(
        self, provider: RealizedFutrProvider, window: Window, hourly_panel: Panel
    ) -> None:
        futr = provider.futr(window, ids=hourly_panel.ids())

        assert len(futr.df) == len(hourly_panel.ids()) * window.h
        assert futr.df["ds"].min() == window.first_pred
        assert futr.df["ds"].max() == window.last_pred
        assert (futr.df["ds"] > window.cutoff).all()

    def test_los_valores_son_los_realizados_del_panel(
        self, provider: RealizedFutrProvider, window: Window, hourly_panel: Panel
    ) -> None:
        futr = provider.futr(window, ids=hourly_panel.ids())
        expected = hourly_panel.df.merge(futr.df, on=["unique_id", "ds"], suffixes=("", "_futr"))
        pd.testing.assert_series_equal(
            expected["temp_c"], expected["temp_c_futr"], check_names=False
        )

    def test_solo_devuelve_las_series_pedidas(
        self, provider: RealizedFutrProvider, window: Window
    ) -> None:
        futr = provider.futr(window, ids=["s00"])
        assert set(futr.df["unique_id"]) == {"s00"}
        assert len(futr.df) == window.h

    def test_lleva_la_ventana_y_el_vintage(
        self, provider: RealizedFutrProvider, window: Window, hourly_panel: Panel
    ) -> None:
        futr = provider.futr(window, ids=hourly_panel.ids())
        assert futr.window == window
        assert futr.vintage is Vintage.REALIZED

    def test_rechaza_cubrir_un_tramo_incompleto(self, hourly_panel: Panel, window: Window) -> None:
        # Entregar menos filas de las debidas dejaria al modelo prediciendo sobre
        # un horizonte distinto del que se evalua, y el merge posterior lo
        # disimularia convirtiendo el desajuste en NaN.
        recortado = hourly_panel.slice(hourly_panel.first_ds, window.first_pred)
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=recortado)

        with pytest.raises(PanelValidationError, match="filas de exogenas futuras"):
            provider.futr(window, ids=hourly_panel.ids())
