"""`request_with_retries`: reintentos, backoff, timeout explicito. Sin red: `httpx.MockTransport`."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from chronolab.data.sources._http import request_with_retries
from chronolab.errors import SourceUnavailable


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _sequenced_client(responses: list[httpx.Response]) -> tuple[httpx.Client, list[httpx.Request]]:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return responses[len(received) - 1]

    return _client(handler), received


class TestExito:
    def test_una_respuesta_200_se_devuelve_sin_reintentar(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json={"ok": True})

        response = request_with_retries(_client(handler), "GET", "http://testserver/x")
        assert response.status_code == 200
        assert len(calls) == 1


class TestReintentos:
    def test_reintenta_tras_un_500_y_termina_en_exito(self) -> None:
        client, received = _sequenced_client(
            [httpx.Response(500), httpx.Response(200, json={"ok": True})]
        )
        response = request_with_retries(client, "GET", "http://testserver/x", backoff_base=0.0)
        assert response.status_code == 200
        assert len(received) == 2

    def test_reintenta_tras_un_429_y_termina_en_exito(self) -> None:
        client, received = _sequenced_client(
            [httpx.Response(429), httpx.Response(200, json={"ok": True})]
        )
        response = request_with_retries(client, "GET", "http://testserver/x", backoff_base=0.0)
        assert response.status_code == 200
        assert len(received) == 2

    def test_reintenta_tras_un_error_de_red_y_termina_en_exito(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("simulado", request=request)
            return httpx.Response(200, json={"ok": True})

        response = request_with_retries(
            _client(handler), "GET", "http://testserver/x", backoff_base=0.0
        )
        assert response.status_code == 200
        assert len(calls) == 2

    def test_agota_reintentos_y_lanza_source_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        with pytest.raises(SourceUnavailable, match="sin respuesta valida"):
            request_with_retries(
                _client(handler), "GET", "http://testserver/x", max_retries=2, backoff_base=0.0
            )

    def test_el_numero_de_intentos_es_max_retries_mas_uno(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503)

        with pytest.raises(SourceUnavailable):
            request_with_retries(
                _client(handler), "GET", "http://testserver/x", max_retries=3, backoff_base=0.0
            )
        assert len(calls) == 4


class TestErroresNoTransitorios:
    def test_un_404_se_propaga_de_inmediato_sin_reintentar(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(404)

        with pytest.raises(httpx.HTTPStatusError):
            request_with_retries(_client(handler), "GET", "http://testserver/x", max_retries=3)
        assert len(calls) == 1

    def test_un_400_se_propaga_de_inmediato_sin_reintentar(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400)

        with pytest.raises(httpx.HTTPStatusError):
            request_with_retries(_client(handler), "GET", "http://testserver/x")


class TestTimeoutExplicito:
    def test_el_timeout_pedido_se_aplica_a_cada_intento(self) -> None:
        seen_timeouts = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_timeouts.append(request.extensions["timeout"]["connect"])
            return httpx.Response(200, json={"ok": True})

        request_with_retries(_client(handler), "GET", "http://testserver/x", timeout=7.5)
        assert seen_timeouts == [7.5]

    def test_nunca_depende_del_timeout_por_defecto_del_cliente(self) -> None:
        # El cliente se crea sin timeout propio; si `request_with_retries` no
        # pasara `timeout` explicito en cada llamada, httpx aplicaria su
        # propio valor por defecto en lugar del pedido.
        seen_timeouts = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_timeouts.append(request.extensions["timeout"]["read"])
            return httpx.Response(200, json={"ok": True})

        client = httpx.Client(transport=httpx.MockTransport(handler), timeout=None)
        request_with_retries(client, "GET", "http://testserver/x", timeout=3.0)
        assert seen_timeouts == [3.0]


class TestParametrosDeQuery:
    def test_los_parametros_llegan_a_la_peticion(self) -> None:
        received_params = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received_params.update(dict(request.url.params))
            return httpx.Response(200, json={"ok": True})

        request_with_retries(
            _client(handler),
            "GET",
            "http://testserver/x",
            params={"symbol": "BTCUSDT", "limit": 100},
        )
        assert received_params == {"symbol": "BTCUSDT", "limit": "100"}
