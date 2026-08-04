"""Normalizacion temporal: DST con fechas reales, rejilla completa, dedup y resample.

El cambio de hora europeo es la trampa clasica en series horarias: si no se
maneja, la estacionalidad diaria se desalinea dos veces al ano sin ningun
sintoma visible. Estos tests usan fechas de transicion reales, no simuladas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronolab.data.align import deduplicate, reindex_to_full_grid, resample_mean, to_utc_naive


class TestToUtcNaivePassthrough:
    def test_fuente_ya_en_utc_no_se_toca(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-06-01 10:00", "2024-06-01 11:00"]))
        result = to_utc_naive(ds, source_tz="UTC")
        pd.testing.assert_series_equal(result, ds)

    def test_entrada_ya_utc_tz_aware_se_despoja_el_huso(self) -> None:
        # Como en REE: cada string trae su propio offset y se resuelve al
        # parsear con utc=True (pandas no admite offsets mixtos sin ese
        # parametro); to_utc_naive solo tiene que confirmar y despojar.
        parsed_utc = pd.to_datetime(
            pd.Series(["2024-01-15T00:00:00+01:00", "2024-07-15T00:00:00+02:00"]), utc=True
        )
        result = to_utc_naive(parsed_utc)
        assert result.dt.tz is None
        assert result.tolist() == [
            pd.Timestamp("2024-01-14 23:00:00"),
            pd.Timestamp("2024-07-14 22:00:00"),
        ]

    def test_entrada_tz_aware_europe_madrid_respeta_el_offset_de_cada_fila(self) -> None:
        # Una serie ya localizada (no naive) en una zona IANA: pandas resuelve
        # el offset correcto por fila segun la fecha, incluso a traves del
        # cambio de hora, sin pasar por la logica de ambiguedad de
        # to_utc_naive (que solo aplica a la rama naive).
        naive = pd.Series(pd.to_datetime(["2024-01-15 12:00", "2024-07-15 12:00"]))
        localized = naive.dt.tz_localize("Europe/Madrid")
        result = to_utc_naive(localized)
        assert result.tolist() == [
            pd.Timestamp("2024-01-15 11:00:00"),  # invierno, UTC+1
            pd.Timestamp("2024-07-15 10:00:00"),  # verano, UTC+2
        ]


class TestSaltoDePrimavera:
    """Ultimo domingo de marzo: 02:00 -> 03:00. La hora 02:xx local no existe."""

    def test_madrid_2024_marca_nat_la_hora_inexistente(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-03-31 01:30", "2024-03-31 02:30", "2024-03-31 03:30"]))
        result = to_utc_naive(ds, source_tz="Europe/Madrid")
        assert result.iloc[0] == pd.Timestamp("2024-03-31 00:30:00")
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pd.Timestamp("2024-03-31 01:30:00")

    def test_madrid_2023_otra_fecha_real_de_transicion(self) -> None:
        # La transicion cae el ultimo domingo de marzo cada ano: en 2023 fue
        # el 26. Confirma que el manejo no esta atado a una fecha concreta.
        ds = pd.Series(pd.to_datetime(["2023-03-26 02:30"]))
        result = to_utc_naive(ds, source_tz="Europe/Madrid")
        assert pd.isna(result.iloc[0])

    def test_lisboa_misma_fecha_de_transicion_que_madrid(self) -> None:
        # Todo el bloque UE cambia de hora el mismo dia civil, aunque el
        # desplazamiento base (UTC+0/+1 en Lisboa frente a UTC+1/+2 en
        # Madrid) sea distinto.
        ds = pd.Series(pd.to_datetime(["2024-03-31 01:30"]))
        result = to_utc_naive(ds, source_tz="Europe/Lisbon")
        assert pd.isna(result.iloc[0])

    def test_las_horas_fuera_de_la_transicion_no_se_alteran(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-03-31 00:00", "2024-04-01 12:00"]))
        result = to_utc_naive(ds, source_tz="Europe/Madrid")
        assert not result.isna().any()


class TestVuelcoDeOtono:
    """Ultimo domingo de octubre: 03:00 -> 02:00. La hora 02:xx local ocurre dos veces."""

    def test_madrid_2024_las_dos_ocurrencias_resuelven_a_instantes_utc_distintos(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-10-27 02:30", "2024-10-27 02:30"]))
        result = to_utc_naive(ds, source_tz="Europe/Madrid")
        assert result.tolist() == [
            pd.Timestamp("2024-10-27 00:30:00"),  # primera: horario de verano, UTC+2
            pd.Timestamp("2024-10-27 01:30:00"),  # segunda: horario de invierno, UTC+1
        ]

    def test_la_separacion_entre_ocurrencias_es_de_una_hora(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-10-27 02:00", "2024-10-27 02:00"]))
        result = to_utc_naive(ds, source_tz="Europe/Madrid")
        assert result.iloc[1] - result.iloc[0] == pd.Timedelta(hours=1)

    def test_madrid_2023_otra_fecha_real_de_transicion(self) -> None:
        # Ultimo domingo de octubre de 2023: el 29.
        ds = pd.Series(pd.to_datetime(["2023-10-29 02:15", "2023-10-29 02:15"]))
        result = to_utc_naive(ds, source_tz="Europe/Madrid")
        assert result.iloc[0] != result.iloc[1]

    def test_sin_group_una_lectura_ajena_desplaza_la_desambiguacion_de_otra_serie(self) -> None:
        # Escenario real: la serie 'b' aporta una lectura no relacionada en la
        # misma hora local ambigua, justo antes que las dos lecturas genuinas
        # de la serie 'a'. Sin `group`, el orden de aparicion combinado hace
        # que las dos lecturas de 'a' -que son instantes reales distintos,
        # separados una hora- colapsen al mismo instante UTC.
        ds = pd.Series(pd.to_datetime(["2024-10-27 02:30", "2024-10-27 02:30", "2024-10-27 02:30"]))
        result = to_utc_naive(ds, source_tz="Europe/Madrid")
        assert result.iloc[1] == result.iloc[2]  # deberian ser distintos: bug

    def test_con_group_las_dos_lecturas_reales_de_la_serie_se_mantienen_distintas(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-10-27 02:30", "2024-10-27 02:30", "2024-10-27 02:30"]))
        group = pd.Series(["b", "a", "a"])
        result = to_utc_naive(ds, source_tz="Europe/Madrid", group=group)
        assert result.iloc[1] == pd.Timestamp("2024-10-27 00:30:00")
        assert result.iloc[2] == pd.Timestamp("2024-10-27 01:30:00")
        assert result.iloc[1] != result.iloc[2]


class TestReindexToFullGrid:
    def test_rellena_huecos_con_nan_explicito(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a"],
                "ds": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"]),
                "y": [1.0, 2.0, 4.0],
            }
        )
        result = reindex_to_full_grid(frame, freq="h")
        assert len(result) == 4
        assert result["y"].tolist()[:2] == [1.0, 2.0]
        assert np.isnan(result["y"].iloc[2])
        assert result["y"].iloc[3] == 4.0

    def test_cada_serie_cubre_solo_su_propio_rango(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b", "b"],
                "ds": pd.to_datetime(
                    [
                        "2024-01-01 00:00",
                        "2024-01-01 01:00",
                        "2024-01-01 05:00",
                        "2024-01-01 06:00",
                        "2024-01-01 07:00",
                    ]
                ),
                "y": [1.0, 2.0, 5.0, 6.0, 7.0],
            }
        )
        result = reindex_to_full_grid(frame, freq="h")
        by_series = result.groupby("unique_id")["ds"]
        assert by_series.get_group("a").tolist() == list(
            pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"])
        )
        assert len(by_series.get_group("b")) == 3

    def test_sin_huecos_es_practicamente_un_no_op(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a"],
                "ds": pd.date_range("2024-01-01", periods=3, freq="h"),
                "y": [1.0, 2.0, 3.0],
            }
        )
        result = reindex_to_full_grid(frame, freq="h")
        assert result["y"].tolist() == [1.0, 2.0, 3.0]

    def test_conserva_las_columnas_de_valor_declaradas(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"]),
                "y": [1.0, 3.0],
                "temp_c": [10.0, 12.0],
            }
        )
        result = reindex_to_full_grid(frame, freq="h")
        assert list(result.columns) == ["unique_id", "ds", "y", "temp_c"]
        assert np.isnan(result["temp_c"].iloc[1])


class TestDeduplicate:
    def _frame_con_duplicado(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unique_id": ["a", "a", "b"],
                "ds": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:00"]),
                "y": [10.0, 20.0, 5.0],
            }
        )

    def test_policy_mean_promedia_los_duplicados(self) -> None:
        result = deduplicate(self._frame_con_duplicado(), policy="mean")
        row_a = result[result["unique_id"] == "a"]
        assert row_a["y"].iloc[0] == pytest.approx(15.0)
        assert len(result) == 2

    def test_policy_first_conserva_la_primera_fila(self) -> None:
        result = deduplicate(self._frame_con_duplicado(), policy="first")
        row_a = result[result["unique_id"] == "a"]
        assert row_a["y"].iloc[0] == 10.0

    def test_policy_last_conserva_la_ultima_fila(self) -> None:
        result = deduplicate(self._frame_con_duplicado(), policy="last")
        row_a = result[result["unique_id"] == "a"]
        assert row_a["y"].iloc[0] == 20.0

    def test_sin_duplicados_no_cambia_nada(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "b"],
                "ds": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00"]),
                "y": [1.0, 2.0],
            }
        )
        result = deduplicate(frame, policy="mean")
        assert len(result) == 2

    def test_la_deduplicacion_es_por_pareja_unique_id_ds(self) -> None:
        # Dos series distintas pueden compartir timestamp sin ser duplicados
        # entre si: cada (unique_id, ds) se evalua de forma independiente.
        result = deduplicate(self._frame_con_duplicado(), policy="mean")
        assert set(result["unique_id"]) == {"a", "b"}


class TestResampleMean:
    def test_promedia_cuatro_cuartos_de_hora_en_una_hora(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 4,
                "ds": pd.date_range("2024-01-01 00:00", periods=4, freq="15min"),
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = resample_mean(frame, freq="h")
        assert len(result) == 1
        assert result["y"].iloc[0] == pytest.approx(2.5)

    def test_un_bucket_horario_sin_observaciones_queda_en_nan(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 4,
                "ds": pd.to_datetime(
                    ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 02:00", "2024-01-01 02:15"]
                ),
                "y": [1.0, 1.0, 3.0, 3.0],
            }
        )
        result = resample_mean(frame, freq="h")
        assert len(result) == 3  # horas 00, 01, 02
        assert np.isnan(result["y"].iloc[1])

    def test_conserva_series_independientes(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "ds": pd.date_range("2024-01-01 00:00", periods=2, freq="30min").tolist() * 2,
                "y": [1.0, 3.0, 10.0, 20.0],
            }
        )
        result = resample_mean(frame, freq="h")
        assert set(result["unique_id"]) == {"a", "b"}
        assert result.set_index("unique_id")["y"].to_dict() == {"a": 2.0, "b": 15.0}
