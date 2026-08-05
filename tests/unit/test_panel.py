"""Contrato del panel: validacion de `PanelSpec` e invariantes de `Panel`."""

from __future__ import annotations

import pandas as pd
import pytest

from chronolab.errors import PanelValidationError
from chronolab.evaluation.splitters import RollingOriginSplitter, Window
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId


def _split(panel: Panel) -> tuple[Window, ...]:
    return RollingOriginSplitter(h=24, n_windows=3, step_size=24).split(panel)


def _spec(**overrides: object) -> PanelSpec:
    kwargs: dict[str, object] = {
        "dataset_id": DatasetId("d"),
        "freq": "h",
        "seasonalities": (24, 168),
    }
    kwargs.update(overrides)
    return PanelSpec(**kwargs)  # type: ignore[arg-type]


class TestPanelSpecValidation:
    def test_acepta_una_especificacion_coherente(self) -> None:
        spec = _spec(futr_exog=("temp_c",), hist_exog=("voltage",))
        assert spec.target == "y"
        assert spec.tz_display == "UTC"

    def test_rechaza_seasonalities_vacias(self) -> None:
        with pytest.raises(PanelValidationError, match="vacia"):
            _spec(seasonalities=())

    @pytest.mark.parametrize("seasonalities", [(1, 24), (0,), (-24,)])
    def test_rechaza_seasonalities_menores_que_dos(self, seasonalities: tuple[int, ...]) -> None:
        with pytest.raises(PanelValidationError, match=">= 2"):
            _spec(seasonalities=seasonalities)

    @pytest.mark.parametrize("seasonalities", [(168, 24), (24, 24), (24, 168, 168)])
    def test_rechaza_seasonalities_no_crecientes(self, seasonalities: tuple[int, ...]) -> None:
        with pytest.raises(PanelValidationError, match="creciente"):
            _spec(seasonalities=seasonalities)

    def test_rechaza_columna_en_dos_roles(self) -> None:
        # El caso peligroso: una exogena declarada a la vez como conocida a
        # futuro y como historica dejaria indefinido si puede leerse tras el
        # cutoff, que es exactamente la ambiguedad que produce fuga.
        with pytest.raises(PanelValidationError, match="dos veces"):
            _spec(futr_exog=("temp_c",), hist_exog=("temp_c",))

    def test_rechaza_exogena_que_colisiona_con_el_objetivo(self) -> None:
        with pytest.raises(PanelValidationError, match="dos veces"):
            _spec(futr_exog=("y",))

    @pytest.mark.parametrize("reserved", ["unique_id", "ds"])
    def test_rechaza_nombres_reservados(self, reserved: str) -> None:
        with pytest.raises(PanelValidationError, match="clave del panel"):
            _spec(hist_exog=(reserved,))


class TestPanelSpecProperties:
    def test_mase_season_es_la_estacionalidad_mas_corta(self) -> None:
        assert _spec(seasonalities=(24, 168, 8766)).mase_season == 24

    def test_value_columns_ordena_objetivo_futuras_e_historicas(self) -> None:
        spec = _spec(futr_exog=("temp_c", "is_holiday"), hist_exog=("voltage",))
        assert spec.value_columns == ("y", "temp_c", "is_holiday", "voltage")

    def test_columns_anade_las_claves_del_panel(self) -> None:
        spec = _spec(futr_exog=("temp_c",))
        assert spec.columns == ("unique_id", "ds", "y", "temp_c")

    def test_las_estaticas_no_son_columnas_del_panel(self) -> None:
        # Viven en la trama lateral `static`, una fila por serie.
        spec = _spec(static_exog=("client_type",))
        assert "client_type" not in spec.columns


class TestPanel:
    def test_ids_devuelve_las_series_sin_repeticiones(self, hourly_panel: Panel) -> None:
        assert hourly_panel.ids() == ("s00", "s01", "s02")

    def test_es_inmutable(self, hourly_panel: Panel) -> None:
        with pytest.raises(AttributeError):
            hourly_panel.spec = _spec()  # type: ignore[misc]

    def test_no_expone_preprocesado_global(self, hourly_panel: Panel) -> None:
        # La ausencia de estos metodos es la barrera contra ajustar un escalador
        # con datos posteriores al cutoff (docs/ARCHITECTURE.md fuga L2).
        for forbidden in ("scale", "impute", "transform", "fillna", "normalize"):
            assert not hasattr(hourly_panel, forbidden)


class TestPanelProjections:
    def test_la_rejilla_cubre_el_panel_a_su_frecuencia(self, hourly_panel: Panel) -> None:
        grid = hourly_panel.grid()
        assert grid[0] == hourly_panel.first_ds
        assert grid[-1] == hourly_panel.last_ds
        assert len(grid) == hourly_panel.df["ds"].nunique()

    def test_slice_recorta_por_extremos_inclusivos(self, hourly_panel: Panel) -> None:
        start = hourly_panel.first_ds + pd.Timedelta(hours=10)
        end = start + pd.Timedelta(hours=5)
        sliced = hourly_panel.slice(start, end)

        assert sliced.first_ds == start
        assert sliced.last_ds == end
        assert len(sliced.df) == 6 * len(hourly_panel.ids())
        assert sliced.spec is hourly_panel.spec

    def test_slice_rechaza_extremos_invertidos(self, hourly_panel: Panel) -> None:
        with pytest.raises(PanelValidationError, match="posterior a end"):
            hourly_panel.slice(hourly_panel.last_ds, hourly_panel.first_ds)

    def test_slice_recorta_tambien_las_estaticas(self, hourly_panel: Panel) -> None:
        # Si `static` conservase series que ya no estan en el panel, un join
        # posterior multiplicaria filas en silencio (invariante I7).
        static = pd.DataFrame({"unique_id": ["s00", "s01", "s02"], "region": ["n", "s", "e"]})
        panel = Panel(df=hourly_panel.df, spec=hourly_panel.spec, static=static)
        only_first = panel.df[panel.df["unique_id"] == "s00"]

        sliced = Panel(df=only_first, spec=panel.spec, static=static).slice(
            panel.first_ds, panel.last_ds
        )
        assert sliced.static is not None
        assert sliced.static["unique_id"].tolist() == ["s00"]

    def test_train_llega_exactamente_al_cutoff(self, hourly_panel: Panel) -> None:
        window = _split(hourly_panel)[0]
        train = hourly_panel.train(window)

        assert train.first_ds == window.train_start
        assert train.last_ds == window.cutoff
        # Lo que el modelo recibe no contiene ni un instante del tramo evaluado.
        assert train.df["ds"].max() < window.first_pred

    def test_actuals_cubre_el_tramo_evaluado(self, hourly_panel: Panel) -> None:
        window = _split(hourly_panel)[0]
        actuals = hourly_panel.actuals(window)

        assert list(actuals.columns) == ["unique_id", "ds", "y"]
        assert actuals["ds"].min() == window.first_pred
        assert actuals["ds"].max() == window.last_pred
        assert len(actuals) == window.h * len(hourly_panel.ids())

    def test_to_nixtla_devuelve_una_copia_en_orden_declarado(self, hourly_panel: Panel) -> None:
        view = hourly_panel.to_nixtla()

        assert list(view.columns) == list(hourly_panel.spec.columns)
        view.loc[0, "y"] = -999.0
        assert hourly_panel.df.loc[0, "y"] != -999.0
