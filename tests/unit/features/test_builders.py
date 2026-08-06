"""Conjuntos de features con nombre: calendario, termicas y su composicion."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chronolab.features import builders
from chronolab.features.ops import Feature
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId

N_HOURS = 24 * 10


def _panel(*, temp_role: str = "futr_exog") -> Panel:
    """Panel horario de una serie, con `temp_c` en el rol pedido."""
    index = pd.date_range("2023-12-28", periods=N_HOURS, freq="h")  # cruza Ano Nuevo
    rng = np.random.default_rng(0)
    temp = 12.0 + 8.0 * np.sin(2 * np.pi * (index.hour - 4) / 24) + rng.normal(0, 0.5, N_HOURS)
    frame = pd.DataFrame(
        {
            "unique_id": "s0",
            "ds": index,
            "y": np.arange(N_HOURS, dtype=float),
            "temp_c": temp.astype(np.float32),
        }
    )
    kwargs = {"futr_exog": ("temp_c",)} if temp_role == "futr_exog" else {"hist_exog": ("temp_c",)}
    spec = PanelSpec(
        dataset_id=DatasetId("mini"),
        freq="h",
        seasonalities=(24,),
        tz_display="Europe/Madrid",
        **kwargs,  # type: ignore[arg-type]
    )
    return Panel(df=frame, spec=spec)


class TestCalendarFeatureSet:
    def test_nombres_esperados_sin_pais(self) -> None:
        features = builders.calendar_feature_set(_panel())
        names = {f.name for f in features}
        expected = {
            "hour",
            "dayofweek",
            "day",
            "month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "fourier_daily_sin_1",
            "fourier_daily_cos_1",
            "fourier_daily_sin_2",
            "fourier_daily_cos_2",
            "fourier_weekly_sin_1",
            "fourier_weekly_cos_1",
            "fourier_annual_sin_1",
            "fourier_annual_cos_1",
        }
        assert expected.issubset(names)
        assert "is_holiday" not in names
        assert "is_holiday_eve" not in names

    def test_anade_festivos_solo_con_pais(self) -> None:
        features = builders.calendar_feature_set(_panel(), country="ES")
        names = {f.name for f in features}
        assert "is_holiday" in names
        assert "is_holiday_eve" in names

    def test_todas_las_features_de_calendario_son_ilimitadas(self) -> None:
        features = builders.calendar_feature_set(_panel(), country="ES")
        assert all(math.isinf(f.max_lead) for f in features)

    def test_alineadas_con_el_panel(self) -> None:
        panel = _panel()
        features = builders.calendar_feature_set(panel)
        by_name = {f.name: f for f in features}
        assert (
            by_name["hour"].values.tolist()
            == panel.df["ds"].dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid").dt.hour.tolist()
        )


class TestThermalFeatureSet:
    def test_nombres_y_conteo(self) -> None:
        panel = _panel()
        config = builders.ThermalFeatureConfig(lags=(1, 24))
        features = builders.thermal_feature_set(panel, config)
        names = {f.name for f in features}
        assert names == {
            "temp_c",
            "temp_c_hdd18",
            "temp_c_cdd18",
            "temp_c_lag1",
            "temp_c_lag24",
            "temp_c_hdd18_lag1",
            "temp_c_hdd18_lag24",
            "temp_c_cdd18_lag1",
            "temp_c_cdd18_lag24",
        }
        assert len(features) == 3 * (1 + 2)

    def test_grados_dia_se_calculan_punto_a_punto(self) -> None:
        panel = _panel()
        features = {f.name: f for f in builders.thermal_feature_set(panel)}
        temp = features["temp_c"].values.to_numpy()
        hdd = features["temp_c_hdd18"].values.to_numpy()
        cdd = features["temp_c_cdd18"].values.to_numpy()
        np.testing.assert_allclose(hdd, np.clip(18.0 - temp, 0.0, None), atol=1e-4)
        np.testing.assert_allclose(cdd, np.clip(temp - 18.0, 0.0, None), atol=1e-4)

    def test_temperatura_futura_hace_todo_el_conjunto_ilimitado(self) -> None:
        panel = _panel(temp_role="futr_exog")
        features = builders.thermal_feature_set(panel, builders.ThermalFeatureConfig(lags=(1, 24)))
        assert all(math.isinf(f.max_lead) for f in features)

    def test_temperatura_historica_limita_por_retardo(self) -> None:
        panel = _panel(temp_role="hist_exog")
        features = {
            f.name: f
            for f in builders.thermal_feature_set(
                panel, builders.ThermalFeatureConfig(lags=(1, 24))
            )
        }
        assert features["temp_c"].max_lead == 0
        assert features["temp_c_hdd18"].max_lead == 0
        assert features["temp_c_cdd18"].max_lead == 0
        assert features["temp_c_lag1"].max_lead == 1
        assert features["temp_c_lag24"].max_lead == 24
        assert features["temp_c_hdd18_lag24"].max_lead == 24

    def test_retardo_invalido_se_rechaza(self) -> None:
        with pytest.raises(ValueError, match="lag debe ser >= 1"):
            builders.ThermalFeatureConfig(lags=(0,))

    def test_columna_sin_rol_declarado_falla_claro(self) -> None:
        panel = _panel()
        with pytest.raises(KeyError, match="nope"):
            builders.thermal_feature_set(panel, builders.ThermalFeatureConfig(temp_column="nope"))


class TestFeatureSet:
    def test_combina_calendario_y_termicas_por_defecto(self) -> None:
        panel = _panel()
        combined = builders.feature_set(panel)
        calendar_only = builders.calendar_feature_set(panel)
        thermal_only = builders.thermal_feature_set(panel)
        assert len(combined) == len(calendar_only) + len(thermal_only)

    def test_thermal_none_omite_las_termicas(self) -> None:
        panel = _panel()
        combined = builders.feature_set(panel, thermal=None)
        assert {f.name for f in combined} == {f.name for f in builders.calendar_feature_set(panel)}


class TestTargetFeatureConfig:
    def test_valores_por_defecto(self) -> None:
        config = builders.DEFAULT_TARGET_FEATURES
        assert config.lags == builders.LAGS
        assert config.roll_windows == builders.ROLL_WINDOWS
        assert config.roll_stats == builders.ROLL_STATS

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lags": (0,)},
            {"roll_windows": (0,)},
            {"roll_shift": 0},
            {"diff_lags": (0,)},
            {"pct_change_lags": (0,)},
            {"diff_shift": 0},
        ],
    )
    def test_rechaza_numeros_no_positivos(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            builders.TargetFeatureConfig(**kwargs)  # type: ignore[arg-type]


class TestSelectUsable:
    def _features(self) -> tuple[Feature, ...]:
        panel = _panel(temp_role="hist_exog")
        return builders.thermal_feature_set(panel, builders.ThermalFeatureConfig(lags=(1, 24)))

    def test_filtra_por_adelanto(self) -> None:
        features = self._features()
        at_1 = {f.name for f in builders.select_usable(features, 1)}
        at_24 = {f.name for f in builders.select_usable(features, 24)}
        at_25 = {f.name for f in builders.select_usable(features, 25)}

        assert "temp_c" not in at_1  # max_lead=0: nunca utilizable para lead>=1
        assert "temp_c_lag1" in at_1
        assert "temp_c_lag24" in at_1  # max_lead=24 cubre tambien lead=1
        assert "temp_c_lag1" not in at_24  # max_lead=1 no llega a lead=24
        assert "temp_c_lag24" in at_24
        assert at_25 == set()  # ningun retardo declarado llega a lead=25

    def test_conserva_los_valores_no_solo_los_nombres(self) -> None:
        features = self._features()
        [selected] = [f for f in builders.select_usable(features, 1) if f.name == "temp_c_lag1"]
        assert isinstance(selected, Feature)
        assert len(selected.values) == N_HOURS


class TestFeatureFrame:
    def test_columnas_y_alineado(self) -> None:
        panel = _panel()
        features = builders.calendar_feature_set(panel)
        frame = builders.feature_frame(panel, features)
        assert list(frame["ds"]) == list(panel.df["ds"])
        assert set(frame.columns) == {"unique_id", "ds", *(f.name for f in features)}

    def test_nombres_repetidos_lanzan(self) -> None:
        panel = _panel()
        feature = builders.calendar_feature_set(panel)[0]
        with pytest.raises(ValueError, match="repetidos"):
            builders.feature_frame(panel, (feature, feature))
