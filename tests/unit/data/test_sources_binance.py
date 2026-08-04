"""`BinanceKlinesSource`: paginacion por limite, epoch UTC, reintentos. Sin red."""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import pytest

from chronolab.data.sources.binance import BinanceKlinesSource
from chronolab.errors import SourceUnavailable, VintageNotSupported
from chronolab.types import Role


def _kline(open_time_ms: int, close: float) -> list[Any]:
    return [
        open_time_ms,
        "0",
        "0",
        "0",
        str(close),
        "0",
        open_time_ms + 3_599_999,
        "0",
        0,
        "0",
        "0",
        "0",
    ]


def _client_returning(klines: list[list[Any]], *, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=klines)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _source(client: httpx.Client, **overrides: object) -> BinanceKlinesSource:
    kwargs: dict[str, object] = {"http_client": client, "backoff_base": 0.0, "timeout": 1.0}
    kwargs.update(overrides)
    return BinanceKlinesSource(**kwargs)  # type: ignore[arg-type]


class TestSpec:
    def test_rol_target_sin_dst_que_resolver(self) -> None:
        spec = BinanceKlinesSource().spec
        assert spec.role is Role.TARGET
        assert spec.native_tz == "UTC"

    def test_source_id_incluye_el_intervalo(self) -> None:
        assert BinanceKlinesSource(interval="4h").spec.source_id == "binance_klines_4h"

    def test_freq_se_deriva_del_intervalo(self) -> None:
        assert BinanceKlinesSource(interval="1h").spec.freq == "h"
        assert BinanceKlinesSource(interval="1d").spec.freq == "D"

    def test_intervalo_no_soportado_lanza_value_error(self) -> None:
        with pytest.raises(ValueError, match="interval no soportado"):
            BinanceKlinesSource(interval="7m")


class TestFetch:
    def test_epoch_ms_se_convierte_a_utc_ingenuo(self) -> None:
        klines = [_kline(1_704_067_200_000, 42_000.0)]  # 2024-01-01T00:00:00Z
        source = _source(_client_returning(klines))
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 01:00")
        )
        assert result["ds"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")
        assert result["y"].iloc[0] == 42_000.0
        assert result["unique_id"].iloc[0] == "BTCUSDT"

    def test_usa_el_precio_de_cierre_no_el_de_apertura(self) -> None:
        row = [1_704_067_200_000, "100", "110", "90", "105", "0", 0, "0", 0, "0", "0", "0"]
        source = _source(_client_returning([row]))
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 01:00")
        )
        assert result["y"].iloc[0] == 105.0

    def test_as_of_no_soportado(self) -> None:
        source = _source(_client_returning([]))
        with pytest.raises(VintageNotSupported):
            source.fetch(
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-02"),
                as_of=pd.Timestamp("2024-01-01"),
            )

    def test_sin_datos_devuelve_trama_vacia_sin_reventar(self) -> None:
        source = _source(_client_returning([]))
        result = source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))
        assert len(result) == 0


class TestPaginacionPorLimite:
    def test_una_pagina_no_llena_no_dispara_una_segunda_peticion(self) -> None:
        klines = [_kline(1_704_067_200_000 + i * 3_600_000, float(i)) for i in range(3)]
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json=klines)

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)))
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 03:00")
        )
        assert len(calls) == 1
        assert len(result) == 3

    def test_una_pagina_llena_avanza_start_time_y_pide_la_siguiente(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Se reduce el limite efectivo de pagina a 2 para poder forzar una
        # "pagina llena" sin fabricar 1000 velas.
        import chronolab.data.sources.binance as binance_module

        monkeypatch.setattr(binance_module, "_MAX_LIMIT", 2)

        first_page = [_kline(1_704_067_200_000 + i * 3_600_000, float(i)) for i in range(2)]
        second_page = [_kline(1_704_067_200_000 + i * 3_600_000, float(i)) for i in range(2, 3)]
        pages = [first_page, second_page]
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json=pages[len(calls) - 1])

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)))
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 03:00")
        )
        assert len(calls) == 2
        assert len(result) == 3
        assert result["y"].tolist() == [0.0, 1.0, 2.0]

    def test_start_time_de_la_query_es_epoch_ms(self) -> None:
        received: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received.update(dict(request.url.params))
            return httpx.Response(200, json=[])

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)))
        source.fetch(start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 01:00"))
        assert received["startTime"] == "1704067200000"

    def test_deduplica_velas_repetidas_en_el_limite_de_dos_paginas(self) -> None:
        # Simula el solape: la ultima vela de una pagina coincide con la
        # primera de la siguiente. Como no se puede forzar una pagina llena
        # de verdad (1000 elementos) en un test, se verifica directamente el
        # comportamiento de deduplicacion sobre el resultado combinado
        # llamando dos veces con rangos solapados y concatenando a mano no
        # aplica aqui: se prueba a traves de _parse + deduplicate.
        from chronolab.data.align import deduplicate

        rows = [_kline(1_704_067_200_000, 1.0), _kline(1_704_067_200_000, 1.0)]
        source = _source(_client_returning(rows))
        frame = source._parse(rows)
        deduped = deduplicate(frame, policy="last")
        assert len(deduped) == 1


class TestReintentosYFallos:
    def test_reintenta_tras_un_fallo_transitorio(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=[_kline(1_704_067_200_000, 1.0)])

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)), max_retries=2)
        result = source.fetch(
            start=pd.Timestamp("2024-01-01 00:00"), end=pd.Timestamp("2024-01-01 01:00")
        )
        assert len(calls) == 2
        assert len(result) == 1

    def test_agota_reintentos_y_lanza_source_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        source = _source(httpx.Client(transport=httpx.MockTransport(handler)), max_retries=1)
        with pytest.raises(SourceUnavailable):
            source.fetch(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))
