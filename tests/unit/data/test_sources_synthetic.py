"""Fuentes sinteticas: estructura, DST genuino, imperfecciones inyectadas y consistencia. Sin red."""

from __future__ import annotations

import pandas as pd
import pytest

from chronolab.data.sources.synthetic import (
    DEMO_SERIES_IDS,
    SyntheticElectricitySource,
    SyntheticWeatherSource,
    _full_grid,
    _synthetic_temperature,
)
from chronolab.errors import VintageNotSupported
from chronolab.types import Role, SeriesId

# Ventana corta para que los tests sean rapidos: la fuente genera siempre su
# rango interno completo y filtra al final, asi que una ventana estrecha no
# reduce el trabajo mucho, pero si el volumen de datos que hay que comparar.
_START = pd.Timestamp("2023-06-01")
_END = pd.Timestamp("2023-06-08")


class TestSyntheticElectricitySourceSpec:
    def test_rol_target(self) -> None:
        assert SyntheticElectricitySource().spec.role is Role.TARGET

    def test_columnas_de_valor(self) -> None:
        assert SyntheticElectricitySource().spec.value_columns == ("y",)

    def test_native_tz_es_madrid(self) -> None:
        # A diferencia del clima, la demanda se genera en hora local con las
        # marcas de DST propias de una fuente real.
        assert SyntheticElectricitySource().spec.native_tz == "Europe/Madrid"


class TestSyntheticElectricitySourceFetch:
    def test_devuelve_las_tres_series_por_defecto(self) -> None:
        result = SyntheticElectricitySource().fetch(start=_START, end=_END)
        assert set(result["unique_id"]) == set(DEMO_SERIES_IDS)

    def test_filtra_por_ids(self) -> None:
        result = SyntheticElectricitySource().fetch(
            start=_START, end=_END, ids=[SeriesId("residential_north")]
        )
        assert set(result["unique_id"]) == {"residential_north"}

    def test_respeta_la_semiapertura_del_rango(self) -> None:
        result = SyntheticElectricitySource().fetch(start=_START, end=_END)
        assert result["ds"].min() >= _START
        assert result["ds"].max() < _END

    def test_as_of_no_soportado(self) -> None:
        with pytest.raises(VintageNotSupported):
            SyntheticElectricitySource().fetch(start=_START, end=_END, as_of=_START)

    def test_no_esta_alineada_puede_traer_duplicados(self) -> None:
        # Documentado explicitamente: esta fuente devuelve la trama cruda a
        # proposito, para que la notebook ejercite align.py de verdad.
        result = SyntheticElectricitySource(seed=0).fetch(
            start=pd.Timestamp("2023-10-20"), end=pd.Timestamp("2023-11-05")
        )
        assert result.duplicated(subset=["unique_id", "ds"]).any()

    def test_el_salto_de_primavera_produce_23_marcas_en_la_rejilla_base(self) -> None:
        # Se prueba contra `_full_grid()` directamente, no contra `fetch()`:
        # `fetch()` pasa por la capa de imperfecciones inyectadas, y un
        # duplicado puede caer por azar justo ese dia (esta cubierto aparte
        # en `test_no_esta_alineada_puede_traer_duplicados`), lo que haria
        # este test fragil si contase filas del resultado final.
        _, local_index = _full_grid()
        spring_day = local_index[
            (local_index >= pd.Timestamp("2024-03-31")) & (local_index < pd.Timestamp("2024-04-01"))
        ]
        assert len(spring_day) == 23

    def test_el_vuelco_de_otono_produce_25_marcas_en_la_rejilla_base(self) -> None:
        _, local_index = _full_grid()
        autumn_day = local_index[
            (local_index >= pd.Timestamp("2023-10-29")) & (local_index < pd.Timestamp("2023-10-30"))
        ]
        assert len(autumn_day) == 25

    def test_es_determinista_para_los_mismos_parametros(self) -> None:
        first = SyntheticElectricitySource(seed=3).fetch(start=_START, end=_END)
        second = SyntheticElectricitySource(seed=3).fetch(start=_START, end=_END)
        pd.testing.assert_frame_equal(
            first.sort_values(["unique_id", "ds"]).reset_index(drop=True),
            second.sort_values(["unique_id", "ds"]).reset_index(drop=True),
        )

    def test_semillas_distintas_dan_imperfecciones_distintas(self) -> None:
        first = SyntheticElectricitySource(seed=1).fetch(
            start=pd.Timestamp("2023-06-01"), end=pd.Timestamp("2024-08-01")
        )
        second = SyntheticElectricitySource(seed=2).fetch(
            start=pd.Timestamp("2023-06-01"), end=pd.Timestamp("2024-08-01")
        )
        assert len(first) != len(second) or not first["y"].equals(second["y"])

    def test_ventanas_distintas_coinciden_en_el_solape(self) -> None:
        # Garantia declarada en el docstring: pedir dos rangos distintos del
        # mismo periodo da los mismos valores donde se solapan.
        wide = SyntheticElectricitySource(seed=0).fetch(
            start=pd.Timestamp("2023-07-01"), end=pd.Timestamp("2023-07-10")
        )
        narrow = SyntheticElectricitySource(seed=0).fetch(
            start=pd.Timestamp("2023-07-03"), end=pd.Timestamp("2023-07-05")
        )
        overlap = wide[
            (wide["ds"] >= pd.Timestamp("2023-07-03")) & (wide["ds"] < pd.Timestamp("2023-07-05"))
        ]
        pd.testing.assert_frame_equal(
            overlap.sort_values(["unique_id", "ds"]).reset_index(drop=True),
            narrow.sort_values(["unique_id", "ds"]).reset_index(drop=True),
        )

    def test_commercial_mixed_tiene_un_tramo_de_ceros_inyectado(self) -> None:
        result = SyntheticElectricitySource(seed=0).fetch(
            start=pd.Timestamp("2023-06-01"), end=pd.Timestamp("2024-08-01")
        )
        commercial = result[result["unique_id"] == "commercial_mixed"]
        assert (commercial["y"] == 0.0).sum() > 0

    def test_residential_no_tiene_ceros_inyectados(self) -> None:
        result = SyntheticElectricitySource(seed=0).fetch(
            start=pd.Timestamp("2023-06-01"), end=pd.Timestamp("2024-08-01")
        )
        residential = result[result["unique_id"] == "residential_north"]
        assert (residential["y"] == 0.0).sum() == 0

    def test_los_valores_son_positivos_fuera_del_tramo_de_ceros(self) -> None:
        result = SyntheticElectricitySource(seed=0).fetch(start=_START, end=_END)
        assert (result["y"] > 0).all()


class TestSyntheticWeatherSourceSpec:
    def test_rol_futr_exog(self) -> None:
        spec = SyntheticWeatherSource().spec
        assert spec.role is Role.FUTR_EXOG
        assert spec.value_columns == ("temp_c",)

    def test_native_tz_es_utc(self) -> None:
        # A diferencia de la demanda, el clima se sirve directamente en UTC,
        # como OpenMeteoSource con timezone=UTC: no hay DST que resolver aqui.
        assert SyntheticWeatherSource().spec.native_tz == "UTC"


class TestSyntheticWeatherSourceFetch:
    def test_sin_huecos_ni_duplicados(self) -> None:
        result = SyntheticWeatherSource().fetch(start=_START, end=_END)
        assert not result.duplicated(subset=["unique_id", "ds"]).any()
        assert not result["temp_c"].isna().any()

    def test_as_of_no_soportado(self) -> None:
        with pytest.raises(VintageNotSupported):
            SyntheticWeatherSource().fetch(start=_START, end=_END, as_of=_START)

    def test_rango_de_temperatura_plausible(self) -> None:
        result = SyntheticWeatherSource().fetch(
            start=pd.Timestamp("2023-06-01"), end=pd.Timestamp("2024-08-01")
        )
        assert result["temp_c"].min() > -15.0
        assert result["temp_c"].max() < 45.0


class TestConsistenciaEntreFuentes:
    def test_la_temperatura_es_identica_a_la_usada_internamente(self) -> None:
        # Ambas fuentes comparten `_synthetic_temperature`: la relacion
        # demanda-temperatura de la Fase 3 no es una coincidencia fabricada
        # por separado.
        utc_index, _ = _full_grid()
        expected = _synthetic_temperature(utc_index)

        weather = SyntheticWeatherSource().fetch(
            start=pd.Timestamp("2023-06-01"), end=pd.Timestamp("2024-08-01")
        )
        assert weather["temp_c"].to_numpy() == pytest.approx(expected)
