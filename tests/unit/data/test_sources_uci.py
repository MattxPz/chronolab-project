"""`UCIElectricitySource`: descarga (mockeada), parseo, filtrado y remuestreo. Sin red."""

from __future__ import annotations

import io
import zipfile

import httpx
import pandas as pd
import pytest

from chronolab.data.sources.uci_electricity import UCIElectricitySource
from chronolab.errors import SourceUnavailable, VintageNotSupported
from chronolab.types import Role, SeriesId

# Enero: Europe/Lisbon esta en horario estandar (UTC+0), asi que la hora local
# coincide con UTC y las columnas se pueden razonar directamente.
_CSV_TEXT = (
    ";MT_001;MT_002\n"
    "2024-01-01 00:00:00;10,0;20,0\n"
    "2024-01-01 00:15:00;11,0;21,0\n"
    "2024-01-01 00:30:00;12,0;22,0\n"
    "2024-01-01 00:45:00;13,0;23,0\n"
    "2024-01-01 01:00:00;14,0;24,0\n"
    "2024-01-01 01:15:00;15,0;25,0\n"
    "2024-01-01 01:30:00;16,0;26,0\n"
    "2024-01-01 01:45:00;17,0;27,0\n"
)


def _zip_bytes(csv_text: str = _CSV_TEXT, member_name: str = "LD2011_2014.txt") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member_name, csv_text)
    return buffer.getvalue()


def _client_returning(content: bytes, *, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _flaky_client(content: bytes, *, fail_times: int) -> tuple[httpx.Client, list[int]]:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= fail_times:
            return httpx.Response(503)
        return httpx.Response(200, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def _source(client: httpx.Client, **overrides: object) -> UCIElectricitySource:
    kwargs: dict[str, object] = {"http_client": client, "backoff_base": 0.0, "timeout": 1.0}
    kwargs.update(overrides)
    return UCIElectricitySource(**kwargs)  # type: ignore[arg-type]


class TestSpec:
    def test_rol_target_y_dos_clientes_de_ejemplo(self) -> None:
        spec = UCIElectricitySource().spec
        assert spec.role is Role.TARGET
        assert spec.value_columns == ("y",)
        assert spec.freq == "h"


class TestFetch:
    def test_devuelve_ambas_series_remuestreadas_a_horario(self) -> None:
        client = _client_returning(_zip_bytes())
        source = _source(client)
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 02:00")
        )
        assert set(result["unique_id"]) == {"MT_001", "MT_002"}
        assert len(result) == 4  # 2 series x 2 horas

    def test_la_media_horaria_coincide_con_el_promedio_de_los_cuatro_cuartos(self) -> None:
        client = _client_returning(_zip_bytes())
        source = _source(client)
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 01:00")
        )
        mt001 = result[result["unique_id"] == "MT_001"]
        assert mt001["y"].iloc[0] == pytest.approx((10.0 + 11.0 + 12.0 + 13.0) / 4)

    def test_filtra_por_client_ids_del_constructor(self) -> None:
        client = _client_returning(_zip_bytes())
        source = _source(client, client_ids=("MT_001",))
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 02:00")
        )
        assert set(result["unique_id"]) == {"MT_001"}

    def test_ids_de_fetch_sobrescribe_client_ids_del_constructor(self) -> None:
        client = _client_returning(_zip_bytes())
        source = _source(client, client_ids=("MT_001",))
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"),
            end=pd.Timestamp("2024-01-01 02:00"),
            ids=[SeriesId("MT_002")],
        )
        assert set(result["unique_id"]) == {"MT_002"}

    def test_respeta_la_semiapertura_del_rango(self) -> None:
        client = _client_returning(_zip_bytes())
        source = _source(client)
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 01:00")
        )
        # end es exclusivo: la hora 01:00 no debe aparecer.
        assert result["ds"].max() < pd.Timestamp("2024-01-01 01:00")

    def test_as_of_no_soportado(self) -> None:
        client = _client_returning(_zip_bytes())
        source = _source(client)
        with pytest.raises(VintageNotSupported):
            source.fetch(
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-02"),
                as_of=pd.Timestamp("2024-01-01"),
            )

    def test_encuentra_el_miembro_del_zip_aunque_no_se_llame_como_lo_esperado(self) -> None:
        client = _client_returning(_zip_bytes(member_name="otro_nombre.txt"))
        source = _source(client)
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 02:00")
        )
        assert len(result) > 0


class TestReintentosYFallos:
    def test_reintenta_tras_un_fallo_transitorio_y_termina_en_exito(self) -> None:
        client, calls = _flaky_client(_zip_bytes(), fail_times=1)
        source = _source(client, max_retries=3)
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 02:00")
        )
        assert len(calls) == 2
        assert len(result) > 0

    def test_agota_reintentos_y_lanza_source_unavailable(self) -> None:
        client, _ = _flaky_client(_zip_bytes(), fail_times=999)
        source = _source(client, max_retries=1)
        with pytest.raises(SourceUnavailable):
            source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))
