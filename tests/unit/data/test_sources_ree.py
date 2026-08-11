"""`REEDemandSource`: paginacion por fecha, offsets explicitos, reintentos. Sin red."""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import pytest

from chronolab.data.sources.ree import REEDemandSource
from chronolab.errors import SourceUnavailable, VintageNotSupported
from chronolab.types import Role


def _payload(values: list[dict[str, Any]], *, title: str = "Real") -> dict[str, Any]:
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

    def test_verano_e_invierno_se_resuelven_cada_uno_con_su_offset(self) -> None:
        # +01:00 (invierno) y +02:00 (verano): cada marca trae su propio
        # offset, sin ambiguedad que resolver. Se piden en ventanas angostas
        # y separadas (no una sola de enero a agosto) para que el resample
        # horario no tenga que rellenar seis meses de NaN entre una y otra.
        winter_source = _source(
            _client_returning(
                _payload([{"value": 1.0, "datetime": "2024-01-15T00:00:00.000+01:00"}])
            )
        )
        winter_result = winter_source.fetch(
            start=pd.Timestamp("2024-01-14 23:00"), end=pd.Timestamp("2024-01-15 01:00")
        )
        assert winter_result["ds"].tolist() == [pd.Timestamp("2024-01-14 23:00:00")]
        assert winter_result["y"].tolist() == [1.0]

        summer_source = _source(
            _client_returning(
                _payload([{"value": 2.0, "datetime": "2024-07-15T00:00:00.000+02:00"}])
            )
        )
        summer_result = summer_source.fetch(
            start=pd.Timestamp("2024-07-14 21:00"), end=pd.Timestamp("2024-07-14 23:00")
        )
        assert summer_result["ds"].tolist() == [pd.Timestamp("2024-07-14 22:00:00")]
        assert summer_result["y"].tolist() == [2.0]

    def test_selecciona_la_serie_por_titulo(self) -> None:
        # La API real trae varias series a la vez bajo `included`
        # ("Real", "Prevista", "Programada", "Programada total"); solo
        # "Real" es la demanda observada que expone esta fuente.
        payload = {
            "included": [
                {
                    "attributes": {
                        "title": "Programada",
                        "values": [{"value": 999.0, "datetime": "2024-01-01T00:00:00.000+01:00"}],
                    }
                },
                {
                    "attributes": {
                        "title": "Real",
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


class TestRemuestreoHorario:
    """Resample a horario de la telemetria cruda de `demanda-tiempo-real`.

    El endpoint ignora `time_trunc=hour` y devuelve telemetria cada 5 minutos
    sin importar el parametro (verificado en vivo el 2026-08-11); `fetch` debe
    reducirla a la media horaria que promete `spec.freq`.
    """

    def test_promedia_las_muestras_de_5_minutos_dentro_de_cada_hora(self) -> None:
        payload = _payload(
            [
                {"value": 10.0, "datetime": "2024-01-01T00:00:00.000+01:00"},
                {"value": 20.0, "datetime": "2024-01-01T00:05:00.000+01:00"},
                {"value": 30.0, "datetime": "2024-01-01T00:10:00.000+01:00"},
                {"value": 100.0, "datetime": "2024-01-01T01:00:00.000+01:00"},
                {"value": 200.0, "datetime": "2024-01-01T01:05:00.000+01:00"},
            ]
        )
        source = _source(_client_returning(payload))
        result = source.fetch(
            start=pd.Timestamp("2023-12-31 23:00"), end=pd.Timestamp("2024-01-02")
        )
        assert result["ds"].tolist() == [
            pd.Timestamp("2023-12-31 23:00:00"),
            pd.Timestamp("2024-01-01 00:00:00"),
        ]
        assert result["y"].tolist() == [20.0, 150.0]

    def test_una_hora_sin_muestras_crudas_queda_como_nan_no_como_hueco_silencioso(self) -> None:
        payload = _payload(
            [
                {"value": 10.0, "datetime": "2024-01-01T00:00:00.000+01:00"},
                # 01:00 sin muestras: hueco real de origen.
                {"value": 30.0, "datetime": "2024-01-01T02:00:00.000+01:00"},
            ]
        )
        source = _source(_client_returning(payload))
        result = source.fetch(
            start=pd.Timestamp("2023-12-31 23:00"), end=pd.Timestamp("2024-01-02")
        )
        # 23:00 (10.0), 00:00 (hueco -> NaN), 01:00 (30.0): el resample no
        # debe saltarse la hora intermedia solo porque el payload crudo la
        # omite.
        assert result["ds"].tolist() == [
            pd.Timestamp("2023-12-31 23:00:00"),
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 01:00:00"),
        ]
        assert result["y"].iloc[0] == 10.0
        assert pd.isna(result["y"].iloc[1])
        assert result["y"].iloc[2] == 30.0


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
        # Cada tramo aporta un unico punto crudo, en Jan1/Jan2/Jan3 00:00; el
        # resample rellena cada hora entre el primero y el ultimo dato real
        # (49 = 2 dias completos + 1), sin perder ninguna en el camino. Las
        # horas sin muestra propia quedan en NaN, un hueco de origen legitimo.
        assert len(result) == 49
        assert result.loc[result["y"].notna(), "y"].tolist() == [100.0, 200.0, 300.0]
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
