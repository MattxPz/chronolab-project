"""`CachedSource`: clave por parametros, invalidacion por antigüedad, obsolescencia."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from chronolab.data.cache import CachedSource
from chronolab.data.protocols import SourceSpec
from chronolab.errors import SourceUnavailable, StaleCacheWarning
from chronolab.types import Role, SeriesId


@dataclass
class CountingSource:
    """`DataSource` de prueba que cuenta llamadas y puede fallar a demanda."""

    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    fail_next: bool = False
    value: float = 1.0

    @property
    def spec(self) -> SourceSpec:
        return SourceSpec(source_id="counting", role=Role.TARGET, value_columns=("y",), freq="h")

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if self.fail_next:
            self.fail_next = False
            raise SourceUnavailable("simulado")
        self.calls.append((start, end))
        return pd.DataFrame(
            {
                "unique_id": ["a"],
                "ds": [start],
                "y": [self.value],
            }
        )


def _range() -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")


class TestCachedSource:
    def test_satisface_el_protocolo_datasource_por_estructura(self, tmp_path: Path) -> None:
        from chronolab.data.protocols import DataSource

        cached = CachedSource(inner=CountingSource(), cache_dir=tmp_path)
        assert isinstance(cached, DataSource)

    def test_expone_el_spec_de_la_fuente_envuelta(self, tmp_path: Path) -> None:
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        assert cached.spec == inner.spec

    def test_la_primera_llamada_consulta_la_fuente_real(self, tmp_path: Path) -> None:
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        start, end = _range()
        cached.fetch(start=start, end=end)
        assert len(inner.calls) == 1

    def test_la_segunda_llamada_con_los_mismos_parametros_usa_la_cache(
        self, tmp_path: Path
    ) -> None:
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        start, end = _range()
        cached.fetch(start=start, end=end)
        cached.fetch(start=start, end=end)
        assert len(inner.calls) == 1

    def test_parametros_distintos_producen_entradas_de_cache_distintas(
        self, tmp_path: Path
    ) -> None:
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        start, end = _range()
        cached.fetch(start=start, end=end)
        cached.fetch(start=start, end=end + pd.Timedelta(days=1))
        assert len(inner.calls) == 2

    def test_la_clave_depende_de_ids(self, tmp_path: Path) -> None:
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        start, end = _range()
        cached.fetch(start=start, end=end, ids=[SeriesId("a")])
        cached.fetch(start=start, end=end, ids=[SeriesId("b")])
        assert len(inner.calls) == 2

    def test_el_orden_de_ids_no_cambia_la_clave(self, tmp_path: Path) -> None:
        # La clave se deriva de los ids ordenados: pedir ["a","b"] o ["b","a"]
        # es la misma consulta logica y debe compartir entrada de cache.
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        start, end = _range()
        cached.fetch(start=start, end=end, ids=[SeriesId("a"), SeriesId("b")])
        cached.fetch(start=start, end=end, ids=[SeriesId("b"), SeriesId("a")])
        assert len(inner.calls) == 1

    def test_escribe_bajo_el_subdirectorio_del_source_id(self, tmp_path: Path) -> None:
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        start, end = _range()
        cached.fetch(start=start, end=end)
        written = list((tmp_path / "counting").glob("*.parquet"))
        assert len(written) == 1

    def test_una_entrada_caducada_por_antiguedad_se_refresca(self, tmp_path: Path) -> None:
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path, max_age=timedelta(seconds=0))
        start, end = _range()
        cached.fetch(start=start, end=end)
        time.sleep(0.05)
        cached.fetch(start=start, end=end)
        assert len(inner.calls) == 2

    def test_una_entrada_fresca_no_se_refresca(self, tmp_path: Path) -> None:
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path, max_age=timedelta(days=1))
        start, end = _range()
        cached.fetch(start=start, end=end)
        cached.fetch(start=start, end=end)
        assert len(inner.calls) == 1

    def test_fuente_no_disponible_sin_cache_previa_propaga_el_error(self, tmp_path: Path) -> None:
        inner = CountingSource(fail_next=True)
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        start, end = _range()
        with pytest.raises(SourceUnavailable):
            cached.fetch(start=start, end=end)

    def test_fuente_no_disponible_con_cache_obsoleta_la_sirve_con_aviso(
        self, tmp_path: Path
    ) -> None:
        inner = CountingSource(value=1.0)
        cached = CachedSource(inner=inner, cache_dir=tmp_path, max_age=timedelta(seconds=0))
        start, end = _range()
        cached.fetch(start=start, end=end)  # entrada inicial
        time.sleep(0.05)

        inner.fail_next = True
        with pytest.warns(StaleCacheWarning):
            result = cached.fetch(start=start, end=end)
        assert result["y"].iloc[0] == 1.0

    def test_el_contenido_leido_de_cache_coincide_con_el_original(self, tmp_path: Path) -> None:
        inner = CountingSource(value=42.0)
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        start, end = _range()
        first = cached.fetch(start=start, end=end)
        second = cached.fetch(start=start, end=end)
        pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))

    def test_la_escritura_es_atomica_no_deja_temporales(self, tmp_path: Path) -> None:
        inner = CountingSource()
        cached = CachedSource(inner=inner, cache_dir=tmp_path)
        start, end = _range()
        cached.fetch(start=start, end=end)
        leftover = list(tmp_path.rglob("*.tmp"))
        assert leftover == []
