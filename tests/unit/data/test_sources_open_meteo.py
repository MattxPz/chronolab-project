"""`OpenMeteoSource`: consulta con timezone=UTC, parseo, reintentos. Sin red."""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import pytest

from chronolab.data.sources.open_meteo import OpenMeteoSource
from chronolab.errors import SourceUnavailable, VintageNotSupported
from chronolab.types import Role


def _payload(times: list[str], temps: list[float]) -> dict[str, Any]:
    return {"hourly": {"time": times, "temperature_2m": temps}}


def _client_returning(payload: dict[str, Any], *, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _source(client: httpx.Client, **overrides: object) -> OpenMeteoSource:
    kwargs: dict[str, object] = {
        "latitude": 40.4168,
        "longitude": -3.7038,
        "http_client": client,
        "backoff_base": 0.0,
        "timeout": 1.0,
    }
    kwargs.update(overrides)
    return OpenMeteoSource(**kwargs)  # type: ignore[arg-type]


class TestSpec:
    def test_rol_futr_exog(self) -> None:
        spec = OpenMeteoSource(latitude=40.4, longitude=-3.7).spec
        assert spec.role is Role.FUTR_EXOG
        assert spec.value_columns == ("temp_c",)
        assert spec.native_tz == "UTC"


class TestLocationId:
    def test_se_deriva_de_la_coordenada_si_no_se_indica(self) -> None:
        source = OpenMeteoSource(latitude=40.4168, longitude=-3.7038)
        assert source._location_id == "40.4168_-3.7038"

    def test_se_puede_fijar_explicitamente(self) -> None:
        source = OpenMeteoSource(latitude=40.4168, longitude=-3.7038, location_id="madrid")
        assert source._location_id == "madrid"


class TestFetch:
    def test_las_marcas_ya_en_utc_no_se_alteran(self) -> None:
        payload = _payload(["2024-01-01T00:00", "2024-01-01T01:00"], [5.0, 4.5])
        source = _source(_client_returning(payload))
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 02:00")
        )
        assert result["ds"].tolist() == [
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 01:00:00"),
        ]
        assert result["temp_c"].tolist() == [5.0, 4.5]

    def test_usa_el_location_id_como_unique_id(self) -> None:
        payload = _payload(["2024-01-01T00:00"], [5.0])
        source = _source(_client_returning(payload), location_id="madrid")
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 01:00")
        )
        assert result["unique_id"].tolist() == ["madrid"]

    def test_respeta_la_semiapertura_del_rango(self) -> None:
        payload = _payload(
            ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"], [1.0, 2.0, 3.0]
        )
        source = _source(_client_returning(payload))
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 02:00")
        )
        assert result["ds"].max() < pd.Timestamp("2024-01-01 02:00")
        assert len(result) == 2

    def test_as_of_no_soportado(self) -> None:
        source = _source(_client_returning(_payload([], [])))
        with pytest.raises(VintageNotSupported):
            source.fetch(
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-02"),
                as_of=pd.Timestamp("2024-01-01"),
            )

    def test_end_date_de_la_query_es_el_ultimo_dia_incluido(self) -> None:
        # `end` es exclusivo; Open-Meteo solo entiende fechas, asi que
        # end_date debe ser el dia anterior cuando `end` cae en medianoche.
        received_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received_params.update(dict(request.url.params))
            return httpx.Response(200, json=_payload(["2024-01-01T00:00"], [1.0]))

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)))
        source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-05"))
        assert received_params["start_date"] == "2024-01-01"
        assert received_params["end_date"] == "2024-01-04"

    def test_pide_siempre_timezone_utc(self) -> None:
        received_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received_params.update(dict(request.url.params))
            return httpx.Response(200, json=_payload(["2024-01-01T00:00"], [1.0]))

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)))
        source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))
        assert received_params["timezone"] == "UTC"


class TestReintentosYFallos:
    def test_reintenta_tras_un_fallo_transitorio(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=_payload(["2024-01-01T00:00"], [1.0]))

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)), max_retries=2)
        result = source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))
        assert len(calls) == 2
        assert len(result) == 1

    def test_agota_reintentos_y_lanza_source_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)), max_retries=1)
        with pytest.raises(SourceUnavailable):
            source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))
