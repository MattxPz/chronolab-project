"""Las series sinteticas cumplen el contrato del panel y tienen la estacionalidad esperada.

Si la fixture no tuviera de verdad estacionalidad de 24 y 168, los tests del
arnes pasarian por vacuidad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronolab.panel import Panel
from tests.fixtures.synthetic import DAILY, WEEKLY, autocorrelation, make_hourly_frame


class TestContratoDelPanel:
    def test_las_columnas_son_las_declaradas_por_la_spec(self, hourly_panel: Panel) -> None:
        assert list(hourly_panel.df.columns) == list(hourly_panel.spec.columns)

    def test_ds_es_ingenuo_y_por_contrato_utc(self, hourly_frame: pd.DataFrame) -> None:
        # Invariante I2: una columna tz-aware es error duro, porque mezclar los
        # dos convenios desalinea la estacionalidad diaria dos veces al ano.
        assert hourly_frame["ds"].dt.tz is None

    def test_la_rejilla_es_completa_y_sin_duplicados(self, hourly_frame: pd.DataFrame) -> None:
        # Invariante I3: un hueco es una fila con NaN, nunca una fila ausente.
        for _, group in hourly_frame.groupby("unique_id"):
            expected = pd.date_range(start=group["ds"].min(), end=group["ds"].max(), freq="h")
            assert group["ds"].tolist() == expected.tolist()

    def test_esta_ordenada_por_serie_y_tiempo(self, hourly_frame: pd.DataFrame) -> None:
        # Invariante I4.
        ordered = hourly_frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(hourly_frame, ordered)

    def test_los_valores_son_float32(self, hourly_frame: pd.DataFrame) -> None:
        # Invariante I5: la precision extra no significa nada frente al error de
        # prediccion, y duplicaria el consumo de memoria.
        for column in ("y", "temp_c", "voltage"):
            assert hourly_frame[column].dtype == np.float32


class TestEstacionalidad:
    def test_hay_estacionalidad_diaria(self, single_series_frame: pd.DataFrame) -> None:
        y = single_series_frame["y"].to_numpy(dtype=np.float64)
        # Medio ciclo diario esta en antifase; el ciclo completo, en fase.
        assert autocorrelation(y, DAILY) > autocorrelation(y, DAILY // 2)
        assert autocorrelation(y, DAILY) > 0.5

    def test_hay_estacionalidad_semanal(self, single_series_frame: pd.DataFrame) -> None:
        y = single_series_frame["y"].to_numpy(dtype=np.float64)
        assert autocorrelation(y, WEEKLY) > autocorrelation(y, WEEKLY // 2)


class TestDeterminismo:
    def test_la_misma_semilla_da_la_misma_serie(self) -> None:
        pd.testing.assert_frame_equal(make_hourly_frame(seed=7), make_hourly_frame(seed=7))

    def test_semillas_distintas_dan_series_distintas(self) -> None:
        a = make_hourly_frame(seed=1)["y"].to_numpy()
        b = make_hourly_frame(seed=2)["y"].to_numpy()
        assert not np.array_equal(a, b)
