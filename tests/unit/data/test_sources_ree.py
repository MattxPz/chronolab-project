"""`REEDemandSource`: paginacion por fecha, offsets explicitos, reintentos. Sin red."""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import pytest

from chronolab.data.sources.ree import REEDemandSource
from chronolab.errors import SourceUnavailable, VintageNotSupported
from chronolab.types import Role


def _payload(values: list[dict[str, Any]], *, title: str = "Demanda real") -> dict[str, Any]:
    return {"included": [{"attributes": {"title": title, "values": values}}]}


def _client_returning(payload: dict[str, Any], *, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _source(client: httpx.Client, **overrides: object) -> REEDemandSource:
    kwargs: dict[str, object] = {"http_client": client, "backoff_base": 0.0, "timeout": 1.0}
    kwargs.update(overrides)
    return REEDemandSource(**kwargs)  # type: ignore[arg-type]


class TestSpec:
    def test_rol_target_una_unica_serie(self) -> None:
        spec = REEDemandSource().spec
        assert spec.role is Role.TARGET
        assert spec.value_columns == ("y",)


class TestParseo:
    def test_offset_explicito_se_convierte_a_utc_sin_ambiguedad(self) -> None:
        payload = _payload(
            [
                {"value": 20000.0, "datetime": "2024-01-01T00:00:00.000+01:00"},
                {"value": 21000.0, "datetime": "2024-01-01T01:00:00.000+01:00"},
            ]
        )
        source = _source(_client_returning(payload))
        result = source.fetch(
            start=pd.Timestamp("2023-12-31 23:00"), end=pd.Timestamp("2024-01-01 01:00")
        )
        assert result["ds"].tolist() == [
            pd.Timestamp("2023-12-31 23:00:00"),
            pd.Timestamp("2024-01-01 00:00:00"),
        ]
        assert result["y"].tolist() == [20000.0, 21000.0]

    def test_verano_e_invierno_en_la_misma_respuesta_se_resuelven_cada_uno_con_su_offset(
        self,
    ) -> None:
        # +01:00 (invierno) y +02:00 (verano) en la misma respuesta: cada
        # marca trae su propio offset, sin ambiguedad que resolver.
        payload = _payload(
            [
                {"value": 1.0, "datetime": "2024-01-15T00:00:00.000+01:00"},
                {"value": 2.0, "datetime": "2024-07-15T00:00:00.000+02:00"},
            ]
        )
        source = _source(_client_returning(payload))
        result = source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-08-01"))
        assert result["ds"].tolist() == [
            pd.Timestamp("2024-01-14 23:00:00"),
            pd.Timestamp("2024-07-14 22:00:00"),
        ]

    def test_selecciona_la_serie_por_titulo(self) -> None:
        payload = {
            "included": [
                {
                    "attributes": {
                        "title": "Demanda programada",
                        "values": [{"value": 999.0, "datetime": "2024-01-01T00:00:00.000+01:00"}],
                    }
                },
                {
                    "attributes": {
                        "title": "Demanda real",
                        "values": [{"value": 111.0, "datetime": "2024-01-01T00:00:00.000+01:00"}],
                    }
                },
            ]
        }
        source = _source(_client_returning(payload))
        result = source.fetch(start=pd.Timestamp("2023-12-31"), end=pd.Timestamp("2024-01-02"))
        assert result["y"].tolist() == [111.0]

    def test_serie_no_encontrada_lanza_value_error(self) -> None:
        payload = _payload([], title="Otro titulo")
        source = _source(_client_returning(payload))
        with pytest.raises(ValueError, match="no se encontro la serie"):
            source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))


class TestPaginacionPorRangoDeFechas:
    def test_un_rango_mas_largo_que_max_window_days_se_pagina(self) -> None:
        # Enero: Madrid en horario de invierno (UTC+1), asi que el start_date
        # local de cada tramo, reconvertido con el mismo offset, cae en el
        # instante UTC exacto que abre ese tramo.
        received_start_dates: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            start_date = request.url.params["start_date"]
            received_start_dates.append(start_date)
            value = float(len(received_start_dates)) * 100.0
            payload = _payload([{"value": value, "datetime": f"{start_date}:00.000+01:00"}])
            return httpx.Response(200, json=payload)

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)), max_window_days=1)
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-04 00:00")
        )

        assert len(received_start_dates) == 3  # tres tramos de 1 dia
        assert result["y"].tolist() == [100.0, 200.0, 300.0]
        assert result["ds"].is_monotonic_increasing

    def test_un_rango_corto_no_se_pagina(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            # 01:00+01:00 = 00:00 UTC, dentro del rango pedido.
            return httpx.Response(
                200, json=_payload([{"value": 1.0, "datetime": "2024-01-01T01:00:00.000+01:00"}])
            )

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)))
        result = source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))
        assert len(calls) == 1
        assert len(result) == 1


class TestVintageYReintentos:
    def test_as_of_no_soportado(self) -> None:
        source = _source(_client_returning(_payload([])))
        with pytest.raises(VintageNotSupported):
            source.fetch(
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-02"),
                as_of=pd.Timestamp("2024-01-01"),
            )

    def test_agota_reintentos_y_lanza_source_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)), max_retries=1)
        with pytest.raises(SourceUnavailable):
            source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))
