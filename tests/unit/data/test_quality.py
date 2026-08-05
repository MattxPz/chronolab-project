"""Perfil de calidad: cobertura, huecos, duplicados, ceros, atipicos, continuidad DST."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronolab.data.quality import (
    coverage_report,
    detect_duplicates,
    detect_outliers,
    detect_zeros,
    dst_transition_report,
)


class TestDetectDuplicates:
    def test_devuelve_todas_las_ocurrencias_del_duplicado(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a", "b"],
                "ds": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01"]),
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = detect_duplicates(frame)
        assert len(result) == 2
        assert set(result["y"]) == {1.0, 2.0}

    def test_sin_duplicados_devuelve_vacio(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "b"],
                "ds": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "y": [1.0, 2.0],
            }
        )
        assert detect_duplicates(frame).empty

    def test_series_distintas_no_cuentan_como_duplicado(self) -> None:
        # Comparten ds pero no unique_id: no es un duplicado.
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "b"],
                "ds": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "y": [1.0, 2.0],
            }
        )
        assert detect_duplicates(frame).empty


class TestDetectZeros:
    def test_marca_solo_los_valores_exactamente_cero(self) -> None:
        frame = pd.DataFrame({"unique_id": ["a"] * 3, "y": [0.0, 0.5, -0.0]})
        result = detect_zeros(frame)
        assert len(result) == 2  # 0.0 y -0.0 son ambos == 0

    def test_sin_ceros_devuelve_vacio(self) -> None:
        frame = pd.DataFrame({"unique_id": ["a"] * 3, "y": [1.0, 2.0, 3.0]})
        assert detect_zeros(frame).empty


class TestDetectOutliers:
    def test_marca_un_valor_extremo(self) -> None:
        rng = np.random.default_rng(0)
        normal = rng.normal(100.0, 2.0, 200)
        values = np.concatenate([normal, [500.0]])
        frame = pd.DataFrame({"unique_id": ["a"] * len(values), "y": values})
        result = detect_outliers(frame, z_threshold=4.0)
        assert 500.0 in result["y"].to_numpy()

    def test_no_marca_valores_dentro_de_la_variacion_normal(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(100.0, 2.0, 200)
        frame = pd.DataFrame({"unique_id": ["a"] * len(values), "y": values})
        result = detect_outliers(frame, z_threshold=4.0)
        assert result.empty

    def test_ignora_los_nan(self) -> None:
        frame = pd.DataFrame({"unique_id": ["a"] * 5, "y": [1.0, np.nan, 1.1, 0.9, 1.0]})
        result = detect_outliers(frame, z_threshold=1.0)
        assert not result["y"].isna().any()

    def test_una_serie_constante_no_produce_division_por_cero(self) -> None:
        # MAD = 0 cuando todos los valores son iguales; debe devolver vacio,
        # no lanzar ni devolver NaN/inf.
        frame = pd.DataFrame({"unique_id": ["a"] * 10, "y": [5.0] * 10})
        result = detect_outliers(frame, z_threshold=4.0)
        assert result.empty

    def test_las_series_se_evaluan_por_separado(self) -> None:
        # Un valor tipico de 'b' seria atipico dentro de 'a', y no debe
        # marcarse porque el umbral se calcula por serie.
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 50 + ["b"] * 50,
                "y": list(np.random.default_rng(1).normal(0.0, 1.0, 50))
                + list(np.random.default_rng(2).normal(1000.0, 1.0, 50)),
            }
        )
        result = detect_outliers(frame, z_threshold=4.0)
        assert result.empty


class TestCoverageReport:
    def _raw_and_aligned(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Cruda: 5 horas, con un duplicado explicito de (a, 10:00).
        raw = pd.DataFrame(
            {
                "unique_id": ["a"] * 6,
                "ds": pd.to_datetime(
                    [
                        "2024-01-01 08:00",
                        "2024-01-01 09:00",
                        "2024-01-01 10:00",
                        "2024-01-01 10:00",
                        "2024-01-01 11:00",
                        "2024-01-01 12:00",
                    ]
                ),
                "y": [1.0, 2.0, 3.0, 3.5, 5.0, 6.0],
            }
        )
        # Alineada: rejilla completa 08:00-12:00, con la hora 10:00 como NaN
        # (hueco) y un cero en la de 09:00.
        aligned = pd.DataFrame(
            {
                "unique_id": ["a"] * 5,
                "ds": pd.date_range("2024-01-01 08:00", periods=5, freq="h"),
                "y": [1.0, 0.0, np.nan, 5.0, 6.0],
            }
        )
        return raw, aligned

    def test_una_fila_por_serie(self) -> None:
        raw, aligned = self._raw_and_aligned()
        report = coverage_report(raw, aligned)
        assert len(report) == 1
        assert report["unique_id"].iloc[0] == "a"

    def test_cuenta_duplicados_de_la_trama_cruda(self) -> None:
        raw, aligned = self._raw_and_aligned()
        report = coverage_report(raw, aligned)
        assert report["n_duplicated_pairs"].iloc[0] == 1

    def test_cuenta_huecos_de_la_trama_alineada(self) -> None:
        raw, aligned = self._raw_and_aligned()
        report = coverage_report(raw, aligned)
        assert report["n_gaps"].iloc[0] == 1

    def test_cuenta_ceros(self) -> None:
        raw, aligned = self._raw_and_aligned()
        report = coverage_report(raw, aligned)
        assert report["n_zeros"].iloc[0] == 1

    def test_la_cobertura_es_uno_menos_la_fraccion_de_huecos(self) -> None:
        raw, aligned = self._raw_and_aligned()
        report = coverage_report(raw, aligned)
        # 5 filas esperadas (08:00..12:00), 1 hueco -> cobertura 4/5.
        assert report["coverage"].iloc[0] == pytest.approx(0.8)

    def test_sin_incidencias_la_cobertura_es_completa(self) -> None:
        aligned = pd.DataFrame(
            {
                "unique_id": ["a"] * 3,
                "ds": pd.date_range("2024-01-01", periods=3, freq="h"),
                "y": [1.0, 2.0, 3.0],
            }
        )
        raw = aligned.copy()
        report = coverage_report(raw, aligned)
        assert report["coverage"].iloc[0] == pytest.approx(1.0)
        assert report["n_gaps"].iloc[0] == 0
        assert report["n_duplicated_pairs"].iloc[0] == 0


class TestDstTransitionReport:
    def test_confirma_ausencia_de_huecos_y_duplicados_tras_alinear(self) -> None:
        from chronolab.data.align import deduplicate, reindex_to_full_grid, to_utc_naive

        # Vuelco de otono real: 2024-10-27, Europe/Madrid. La hora local
        # 02:00 aparece dos veces en la trama cruda (una vez en horario de
        # verano, otra en horario de invierno): se anade explicitamente una
        # segunda fila con esa misma etiqueta, igual que haria una fuente real.
        local_ds = pd.date_range("2024-10-26 22:00", "2024-10-27 04:00", freq="h").tolist()
        local_ds += [pd.Timestamp("2024-10-27 02:00")]  # duplicado propio del DST
        raw = pd.DataFrame(
            {"unique_id": "a", "ds": sorted(local_ds), "y": range(len(local_ds))}
        ).astype({"y": "float64"})

        aligned = raw.copy()
        aligned["ds"] = to_utc_naive(
            aligned["ds"], source_tz="Europe/Madrid", group=aligned["unique_id"]
        )
        aligned = aligned.dropna(subset=["ds"])
        aligned = deduplicate(aligned, policy="mean")
        aligned = reindex_to_full_grid(aligned, freq="h")

        report = dst_transition_report(raw, aligned, transitions=[pd.Timestamp("2024-10-27")])
        assert len(report) == 1
        assert report["n_duplicated_ds_aligned"].iloc[0] == 0
        assert report["n_gap_rows_aligned"].iloc[0] == 0

    def test_cuenta_las_filas_locales_del_dia_de_transicion(self) -> None:
        local_ds = pd.date_range("2024-10-27 00:00", "2024-10-27 23:00", freq="h")
        raw = pd.DataFrame({"unique_id": "a", "ds": local_ds, "y": 1.0})
        aligned = pd.DataFrame({"unique_id": "a", "ds": local_ds, "y": 1.0})

        report = dst_transition_report(raw, aligned, transitions=[pd.Timestamp("2024-10-27")])
        assert report["n_rows_local_day"].iloc[0] == 24

    def test_una_fila_por_transicion_pedida(self) -> None:
        raw = pd.DataFrame({"unique_id": ["a"], "ds": [pd.Timestamp("2024-01-01")], "y": [1.0]})
        aligned = raw.copy()
        transitions = [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-10-27")]
        report = dst_transition_report(raw, aligned, transitions=transitions)
        assert len(report) == 2
        assert list(report["transition"]) == transitions
