"""Ayudante interno de red: reintentos con backoff exponencial y timeout explicito.

No es un modulo del arbol publicado en docs/ARCHITECTURE.md §2: es un detalle de
implementacion privado (prefijo `_`), compartido por las tres fuentes que hablan
HTTP (REE, Open-Meteo, Binance) para no repetir la misma logica de reintentos
tres veces. `UCIElectricitySource` tambien lo usa para su descarga del zip.
"""

import time
from collections.abc import Mapping

import httpx

from chronolab.errors import SourceUnavailable

QueryValue = str | int | float | bool | None
"""Tipos de valor de query aceptados por `httpx`, repetidos aqui para no depender
de los tipos privados de `httpx._types`."""

__all__ = [
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT",
    "QueryValue",
    "request_with_retries",
]

DEFAULT_TIMEOUT = 10.0
"""Segundos de espera por intento. Nunca se confia en el timeout por defecto del cliente."""

DEFAULT_MAX_RETRIES = 3
"""Reintentos adicionales tras el primer intento fallido."""

DEFAULT_BACKOFF_BASE = 0.5
"""Base en segundos del backoff exponencial: espera `base * 2**intento` entre reintentos."""


def request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    params: Mapping[str, QueryValue] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
) -> httpx.Response:
    """Ejecuta una peticion HTTP con reintentos y backoff exponencial.

    Solo se reintentan los fallos que tiene sentido reintentar: errores de red
    o de timeout, respuestas ``429`` (limite de tasa) y ``5xx`` (error del
    servidor). Una respuesta ``4xx`` distinta de ``429`` es un error del
    llamante (URL o parametros invalidos) y se propaga de inmediato como
    `httpx.HTTPStatusError`: reintentarla no cambiaria el resultado.

    Parameters
    ----------
    client
        Cliente HTTP ya construido. Se inyecta en lugar de crearse aqui para
        que los tests puedan sustituirlo por uno con
        ``transport=httpx.MockTransport(...)`` y no requerir red.
    method
        Verbo HTTP, por ejemplo ``"GET"``.
    url
        URL completa del recurso.
    params
        Parametros de query.
    timeout
        Segundos de espera por intento individual.
    max_retries
        Numero de reintentos tras el primer fallo. Con el valor por defecto se
        hacen hasta 4 intentos en total.
    backoff_base
        Base del backoff exponencial. Los tests pueden pasar ``0.0`` para que
        los reintentos no introduzcan esperas reales.

    Returns
    -------
    httpx.Response
        La respuesta con codigo de exito.

    Raises
    ------
    SourceUnavailable
        Si se agotan los reintentos sin obtener una respuesta valida.
    httpx.HTTPStatusError
        Si el servidor responde con un error de cliente no transitorio
        (``4xx`` distinto de ``429``).
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.request(method, url, params=params, timeout=timeout)
        except httpx.HTTPError as exc:
            last_error = exc
        else:
            if response.status_code == 429 or response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"{response.status_code} en {url}",
                    request=response.request,
                    response=response,
                )
            else:
                response.raise_for_status()
                return response

        if attempt < max_retries:
            time.sleep(backoff_base * (2**attempt))

    raise SourceUnavailable(
        f"{url}: sin respuesta valida tras {max_retries + 1} intentos"
    ) from last_error
