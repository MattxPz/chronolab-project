"""Fuente Binance klines: serie de contraste donde ningun modelo bate al naive."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from chronolab.data.align import deduplicate
from chronolab.data.protocols import SourceSpec
from chronolab.data.schemas import binance_schema
from chronolab.data.sources._http import (
    DEFAULT_BACKOFF_BASE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    QueryValue,
    request_with_retries,
)
from chronolab.errors import VintageNotSupported
from chronolab.types import Role, SeriesId

__all__ = ["BinanceKlinesSource"]

_DEFAULT_BASE_URL = "https://api.binance.com/api/v3/klines"
_MAX_LIMIT = 1000
"""Maximo de velas por respuesta en la API publica de Binance."""

_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 3_600_000,
    "2h": 2 * 3_600_000,
    "4h": 4 * 3_600_000,
    "1d": 86_400_000,
}
_INTERVAL_FREQ: dict[str, str] = {
    "1m": "min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "h",
    "2h": "2h",
    "4h": "4h",
    "1d": "D",
}


def _empty_klines_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": pd.Series(dtype="object"),
            "ds": pd.Series(dtype="datetime64[ns]"),
            "y": pd.Series(dtype="float64"),
        }
    )


@dataclass(frozen=True, slots=True)
class BinanceKlinesSource:
    """`DataSource` para velas (klines) publicas de Binance.

    Sirve como **serie de contraste**: una serie financiera de alta frecuencia
    es casi un paseo aleatorio, y el baseline naive gana casi siempre.
    Incluirla deliberadamente, y decirlo en el README, es parte del rigor del
    proyecto (docs/PLAN_PROYECTO.md §0).

    El endpoint no tiene DST que resolver: Binance opera siempre en epoch UTC,
    asi que la conversion es aritmetica pura sobre milisegundos y
    `spec.native_tz` es ``"UTC"``.

    Pagina por limite de resultados: cada respuesta trae a lo sumo
    `_MAX_LIMIT` velas (1000, el maximo de la API publica), asi que `fetch`
    avanza ``startTime`` tras cada pagina hasta cubrir ``[start, end)`` o hasta
    que el servidor devuelve menos velas de las pedidas, lo que senala que no
    quedan mas datos en el rango.

    Parameters
    ----------
    symbol
        Par de negociacion, por ejemplo ``"BTCUSDT"``.
    interval
        Intervalo de vela de Binance. Soportados: ``1m, 3m, 5m, 15m, 30m, 1h,
        2h, 4h, 1d``.
    base_url
        URL del endpoint. Parametrizable para tests.
    http_client, timeout, max_retries, backoff_base
        Ver `chronolab.data.sources._http.request_with_retries`.

    Raises
    ------
    ValueError
        Si `interval` no esta en la lista soportada.

    Notes
    -----
    Deduplicacion: ``policy="last"``. Si dos paginas se solapan en el limite
    (la ultima vela de una pagina coincide con la primera de la siguiente), se
    conserva una unica copia; el valor no deberia diferir entre copias porque
    las velas cerradas de Binance no se revisan.
    """

    symbol: str = "BTCUSDT"
    interval: str = "1h"
    base_url: str = _DEFAULT_BASE_URL
    http_client: httpx.Client | None = None
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE

    def __post_init__(self) -> None:
        """Valida que `interval` sea uno de los soportados."""
        if self.interval not in _INTERVAL_MS:
            raise ValueError(
                f"interval no soportado: {self.interval!r}. Usa uno de {sorted(_INTERVAL_MS)}"
            )

    @property
    def spec(self) -> SourceSpec:
        """Fuente de rol `TARGET`: precio de cierre de cada vela."""
        return SourceSpec(
            source_id=f"binance_klines_{self.interval}",
            role=Role.TARGET,
            value_columns=("y",),
            freq=_INTERVAL_FREQ[self.interval],
            native_tz="UTC",
            vintage_aware=False,
            id_semantics="simbolo del par de negociacion (p.ej. BTCUSDT)",
        )

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Descarga velas paginando por limite hasta cubrir `[start, end)`.

        Ver `chronolab.data.protocols.DataSource.fetch`. `as_of` no esta
        soportado: Binance no versiona velas historicas por fecha de consulta.
        """
        if as_of is not None:
            raise VintageNotSupported(
                f"{self.spec.source_id} no admite as_of (no es vintage-aware)"
            )

        client = self.http_client if self.http_client is not None else httpx.Client()
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        step_ms = _INTERVAL_MS[self.interval]

        rows: list[list[Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            page = self._fetch_page(client, cursor, end_ms)
            if not page:
                break
            rows.extend(page)
            cursor = page[-1][0] + step_ms
            if len(page) < _MAX_LIMIT:
                break

        frame = self._parse(rows)
        frame = deduplicate(frame, policy="last")
        frame = frame[(frame["ds"] >= start) & (frame["ds"] < end)]
        frame = frame.sort_values("ds").reset_index(drop=True)

        validated: pd.DataFrame = binance_schema().validate(frame, lazy=True)
        return validated

    def _fetch_page(self, client: httpx.Client, start_ms: int, end_ms: int) -> list[list[Any]]:
        params: dict[str, QueryValue] = {
            "symbol": self.symbol,
            "interval": self.interval,
            "startTime": start_ms,
            # endTime es inclusivo en la API; se resta 1 ms para que el par
            # (startTime, endTime) se comporte como el resto del proyecto:
            # semiabierto por el lado derecho.
            "endTime": end_ms - 1,
            "limit": _MAX_LIMIT,
        }
        response = request_with_retries(
            client,
            "GET",
            self.base_url,
            params=params,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
        )
        klines: list[list[Any]] = response.json()
        return klines

    def _parse(self, rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            return _empty_klines_frame()
        open_time_ms = [row[0] for row in rows]
        close_price = [float(row[4]) for row in rows]
        ds = pd.to_datetime(pd.Series(open_time_ms), unit="ms").astype("datetime64[ns]")
        return pd.DataFrame({"unique_id": self.symbol, "ds": ds, "y": close_price})
