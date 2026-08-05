"""Primitivas retrospectivas: valores correctos y adelanto propagado."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chronolab.features import ops
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId

SPEC = PanelSpec(
    dataset_id=DatasetId("mini"),
    freq="h",
    seasonalities=(2, 4),
    futr_exog=("temp_c",),
    hist_exog=("voltage",),
)


@pytest.fixture
def mini() -> Panel:
    """Dos series de seis horas con valores enteros, calculables a mano."""
    index = pd.date_range("2023-01-02", periods=6, freq="h")
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": index,
                    "y": np.arange(1, 7, dtype=float) * factor,
                    "temp_c": np.arange(10, 16, dtype=float),
                    "voltage": np.arange(100, 106, dtype=float),
                }
            )
            for uid, factor in (("a", 1.0), ("b", 10.0))
        ],
        ignore_index=True,
    )
    return Panel(df=frame, spec=SPEC)


class TestValores:
    def test_lag_desplaza_dentro_de_cada_serie(self, mini: Panel) -> None:
        feature = ops.lag(mini, "y", 2)
        values = feature.values.to_numpy()

        assert feature.name == "y_lag2"
        np.testing.assert_array_equal(values[:6], [np.nan, np.nan, 1, 2, 3, 4])
        # La serie siguiente no arrastra el final de la anterior.
        np.testing.assert_array_equal(values[6:], [np.nan, np.nan, 10, 20, 30, 40])

    def test_diff_resta_el_valor_de_k_pasos_antes(self, mini: Panel) -> None:
        values = ops.diff(mini, "y", 1).values.to_numpy()
        np.testing.assert_array_equal(values[:6], [np.nan, 1, 1, 1, 1, 1])

    def test_roll_termina_en_t_menos_shift(self, mini: Panel) -> None:
        feature = ops.roll(mini, "y", 2, shift=1, stat="mean")
        values = feature.values.to_numpy()

        assert feature.name == "y_rollmean2_s1"
        # En t=2 (valor 3) la ventana cubre {1, 2}: no incluye el propio 3.
        np.testing.assert_array_equal(values[:6], [np.nan, np.nan, 1.5, 2.5, 3.5, 4.5])

    def test_roll_exige_la_ventana_completa(self, mini: Panel) -> None:
        # Un promedio de dos puntos donde deberia haber cuatro no es "casi lo
        # mismo": es otra feature, y encima solo en el arranque de la serie.
        values = ops.roll(mini, "y", 4, shift=1).values.to_numpy()
        assert np.isnan(values[:4]).all()
        assert values[4] == pytest.approx(2.5)

    @pytest.mark.parametrize(
        ("stat", "esperado"),
        [("min", 1.0), ("max", 2.0), ("sum", 3.0), ("median", 1.5), ("std", 0.5**0.5)],
    )
    def test_roll_admite_los_estadisticos_declarados(
        self, mini: Panel, stat: str, esperado: float
    ) -> None:
        values = ops.roll(mini, "y", 2, shift=1, stat=stat).values.to_numpy()  # type: ignore[arg-type]
        # En t=2 la ventana cubre {1, 2}.
        assert values[2] == pytest.approx(esperado)

    def test_expand_acumula_desde_el_inicio_de_la_serie(self, mini: Panel) -> None:
        values = ops.expand(mini, "y", shift=1, stat="mean").values.to_numpy()
        np.testing.assert_allclose(values[:6], [np.nan, 1.0, 1.5, 2.0, 2.5, 3.0])

    def test_ewm_pondera_el_pasado(self, mini: Panel) -> None:
        values = ops.ewm(mini, "y", halflife=1.0, shift=1).values.to_numpy()
        assert np.isnan(values[0])
        assert values[1] == pytest.approx(1.0)
        assert 1.0 < values[2] < 2.0

    def test_lead_sobre_una_futura_adelanta(self, mini: Panel) -> None:
        values = ops.lead(mini, "temp_c", 1).values.to_numpy()
        np.testing.assert_array_equal(values[:6], [11, 12, 13, 14, 15, np.nan])

    def test_las_operaciones_se_componen(self, mini: Panel) -> None:
        media = ops.roll(mini, "y", 2, shift=1)
        feature = ops.lag(mini, media, 1)

        assert feature.name == "y_rollmean2_s1_lag1"
        np.testing.assert_array_equal(
            feature.values.to_numpy()[:6], [np.nan, np.nan, np.nan, 1.5, 2.5, 3.5]
        )


class TestAdelantoPropagado:
    def test_el_lag_de_la_objetivo_caduca_a_los_k_pasos(self, mini: Panel) -> None:
        assert ops.lag(mini, "y", 24).max_lead == 24

    def test_el_lag_de_una_futura_sigue_siendo_ilimitado(self, mini: Panel) -> None:
        assert math.isinf(ops.lag(mini, "temp_c", 3).max_lead)

    def test_la_ventana_movil_hereda_el_desplazamiento(self, mini: Panel) -> None:
        assert ops.roll(mini, "y", 168, shift=6).max_lead == 6
        assert ops.expand(mini, "voltage", shift=2).max_lead == 2

    def test_la_composicion_acumula_adelanto(self, mini: Panel) -> None:
        media = ops.roll(mini, "y", 2, shift=1)  # max_lead 1
        assert ops.lag(mini, media, 23).max_lead == 24

    def test_from_column_toma_el_adelanto_del_rol(self, mini: Panel) -> None:
        assert ops.from_column(mini, "y").max_lead == 0
        assert ops.from_column(mini, "voltage").max_lead == 0
        assert math.isinf(ops.from_column(mini, "temp_c").max_lead)


class TestBarreras:
    def test_no_existe_ninguna_primitiva_prospectiva_sobre_columnas_historicas(
        self, mini: Panel
    ) -> None:
        for column in ("y", "voltage"):
            with pytest.raises(ValueError, match="fuga por construccion"):
                ops.lead(mini, column, 1)

    def test_el_modulo_no_exporta_operaciones_centradas(self) -> None:
        # La barrera es la ausencia de la funcion, no un aviso en la revision.
        for forbidden in ("center", "rolling_center", "future_roll", "shift_forward"):
            assert forbidden not in ops.__all__

    @pytest.mark.parametrize("shift", [0, -3])
    def test_las_ventanas_no_pueden_incluir_el_instante_predicho(
        self, mini: Panel, shift: int
    ) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            ops.roll(mini, "y", 2, shift=shift)

    def test_la_ventana_debe_tener_longitud_positiva(self, mini: Panel) -> None:
        with pytest.raises(ValueError, match="ventana debe ser >= 1"):
            ops.roll(mini, "y", 0)

    def test_la_semivida_debe_ser_positiva(self, mini: Panel) -> None:
        with pytest.raises(ValueError, match="semivida debe ser > 0"):
            ops.ewm(mini, "y", halflife=0.0)

    def test_el_adelanto_debe_ser_positivo(self, mini: Panel) -> None:
        with pytest.raises(ValueError, match="adelanto debe ser >= 1"):
            ops.lead(mini, "temp_c", 0)

    def test_el_estadistico_debe_estar_admitido(self, mini: Panel) -> None:
        with pytest.raises(ValueError, match="estadistico no admitido"):
            ops.roll(mini, "y", 2, stat="mediana_movil")  # type: ignore[arg-type]
