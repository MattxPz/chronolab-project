"""Algebra de `max_lead`: el adelanto se calcula, nunca se declara."""

from __future__ import annotations

import math

import pytest

from chronolab.features.roles import (
    UNBOUNDED,
    FeatureSpec,
    after_diff,
    after_lag,
    after_lead,
    after_roll,
    column_max_lead,
    select_for_lead,
    usable_for_lead,
)
from chronolab.panel import PanelSpec
from chronolab.types import DatasetId

SPEC = PanelSpec(
    dataset_id=DatasetId("d"),
    freq="h",
    seasonalities=(24, 168),
    futr_exog=("temp_c", "is_holiday"),
    hist_exog=("voltage",),
    static_exog=("region",),
)


class TestColumnMaxLead:
    @pytest.mark.parametrize("column", ["y", "voltage"])
    def test_la_objetivo_y_las_historicas_no_se_conocen_en_ningun_futuro(self, column: str) -> None:
        assert column_max_lead(SPEC, column) == 0

    @pytest.mark.parametrize("column", ["temp_c", "is_holiday", "region"])
    def test_las_futuras_y_las_estaticas_se_conocen_en_todo_el_horizonte(self, column: str) -> None:
        assert math.isinf(column_max_lead(SPEC, column))

    def test_una_columna_sin_rol_no_tiene_adelanto_por_defecto(self) -> None:
        # Un valor por defecto convertiria "nadie ha decidido si esto puede
        # usarse" en "puede usarse", que es como acaban colandose las features
        # que nadie autorizo.
        with pytest.raises(KeyError, match="no tiene rol declarado"):
            column_max_lead(SPEC, "columna_fantasma")


class TestPropagacion:
    def test_retrasar_compra_adelanto(self) -> None:
        assert after_lag(0, 24) == 24
        assert after_lag(24, 24) == 48

    def test_sobre_una_conocida_a_futuro_el_retardo_no_cambia_nada(self) -> None:
        assert math.isinf(after_lag(UNBOUNDED, 24))

    def test_la_diferencia_hereda_el_adelanto_del_retardo(self) -> None:
        assert after_diff(0, 24) == after_lag(0, 24)

    def test_la_ventana_movil_depende_de_donde_acaba_no_de_cuanto_abarca(self) -> None:
        # Una media de 168 pasos que termina en t-1 es utilizable a un paso, igual
        # que una de 3 pasos que termina en t-1.
        assert after_roll(0, shift=1) == 1
        assert after_roll(0, shift=24) == 24

    def test_adelantar_una_historica_es_fuga_por_construccion(self) -> None:
        with pytest.raises(ValueError, match="fuga por construccion"):
            after_lead(0, 1)
        with pytest.raises(ValueError, match="fuga por construccion"):
            after_lead(24, 1)

    def test_adelantar_una_conocida_a_futuro_es_legitimo(self) -> None:
        assert math.isinf(after_lead(UNBOUNDED, 6))

    @pytest.mark.parametrize("k", [0, -1])
    def test_los_desplazamientos_deben_ser_positivos(self, k: int) -> None:
        with pytest.raises(ValueError):
            after_lag(0, k)
        with pytest.raises(ValueError):
            after_roll(0, shift=k)


class TestSeleccionPorAdelanto:
    def test_una_feature_caduca_pasado_su_adelanto(self) -> None:
        lag24 = FeatureSpec(name="y_lag24", max_lead=24)
        assert usable_for_lead(lag24, 24)
        # `lag(y, 24)` para predecir a 48 pasos exigiria conocer y en t-24, que
        # en ese momento aun no ha ocurrido: es la fuga L8.
        assert not usable_for_lead(lag24, 25)

    def test_el_motor_filtra_las_features_por_adelanto(self) -> None:
        features = (
            FeatureSpec(name="y_lag1", max_lead=1),
            FeatureSpec(name="y_lag24", max_lead=24),
            FeatureSpec(name="temp_c", max_lead=UNBOUNDED),
        )
        assert [f.name for f in select_for_lead(features, 1)] == [
            "y_lag1",
            "y_lag24",
            "temp_c",
        ]
        assert [f.name for f in select_for_lead(features, 24)] == ["y_lag24", "temp_c"]
        assert [f.name for f in select_for_lead(features, 48)] == ["temp_c"]

    def test_la_estrategia_recursiva_admite_features_caducadas_marcadas(self) -> None:
        features = (
            FeatureSpec(name="y_lag1", max_lead=1, recursive_only=True),
            FeatureSpec(name="y_lag2", max_lead=2),
        )
        assert [f.name for f in select_for_lead(features, 48)] == []
        assert [f.name for f in select_for_lead(features, 48, supports_recursive=True)] == [
            "y_lag1"
        ]

    def test_el_adelanto_declarado_debe_ser_entero_no_negativo(self) -> None:
        with pytest.raises(ValueError, match="max_lead"):
            FeatureSpec(name="rara", max_lead=-1)
        with pytest.raises(ValueError, match="max_lead"):
            FeatureSpec(name="rara", max_lead=2.5)
