"""Fuente REE `apidatos.ree.es`: demanda electrica de Espana, casi tiempo real.

Al ser reciente admite `ARCHIVED_FORECAST` en `chronolab.data.futr`, lo que la
convierte en la serie donde comparar los tres vintages de exogenas futuras.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from chronolab.data.align import deduplicate, to_utc_naive
from chronolab.data.protocols import SourceSpec
from chronolab.data.schemas import ree_demand_schema
from chronolab.data.sources._http import (
    DEFAULT_BACKOFF_BASE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    request_with_retries,
)
from chronolab.errors import VintageNotSupported
from chronolab.types import Role, SeriesId

__all__ = ["REEDemandSource"]

_DEFAULT_BASE_URL = "https://apidatos.ree.es/es/datos/demanda/demanda-tiempo-real"
_NATIVE_TZ = "Europe/Madrid"
_UNIQUE_ID = "ES"
_DEFAULT_MAX_WINDOW_DAYS = 364
"""apidatos.ree.es limita el rango admitido por consulta; se pagina por debajo de eso."""


def _date_chunks(
    start: pd.Timestamp, end: pd.Timestamp, *, max_days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Parte `[start, end)` en tramos consecutivos de a lo sumo `max_days` dias."""
    if start >= end:
        return []
    step = pd.Timedelta(days=max_days)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + step, end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunks


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": pd.Series(dtype="object"),
            "ds": pd.Series(dtype="datetime64[ns]"),
            "y": pd.Series(dtype="float64"),
        }
    )


@dataclass(frozen=True, slots=True)
class REEDemandSource:
    """`DataSource` para la demanda electrica en tiempo real de apidatos.ree.es.

    La API responde en JSON con la forma::

        {
            "included": [
                {
                    "attributes": {
                        "title": "Demanda real",
                        "values": [
                            {"value": 24567.3, "datetime": "2019-08-01T00:00:00.000+02:00"},
                            ...,
                        ],
                    }
                }
            ]
        }

    Cada ``datetime`` trae su propio desplazamiento UTC explicito (``+01:00``
    en horario de invierno, ``+02:00`` en horario de verano): la conversion a
    UTC no tiene ninguna ambiguedad de DST que resolver, porque el offset ya
    esta en el dato. Se parsea con ``utc=True`` directamente y se reexpresa en
    UTC ingenuo con `to_utc_naive`, que en la rama tz-aware se limita a
    confirmar y despojar el huso.

    Como el rango admitido por consulta esta limitado, `fetch` **pagina por
    rango de fechas**: parte ``[start, end)`` en tramos de a lo sumo
    `max_window_days` dias y concatena las respuestas.

    Parameters
    ----------
    base_url
        URL del endpoint. Parametrizable para tests.
    series_title
        Titulo de la serie a extraer de ``included`` (el mismo endpoint puede
        traer mas de una serie, por ejemplo "Demanda real" y
        "Demanda programada").
    max_window_days
        Tamano maximo, en dias, de cada tramo de la paginacion.
    http_client
        Cliente HTTP inyectable. Si es ``None``, se crea uno por defecto.
    timeout, max_retries, backoff_base
        Ver `chronolab.data.sources._http.request_with_retries`.

    Notes
    -----
    Deduplicacion: ``policy="last"``. Si dos tramos de la paginacion se
    solapan en un instante (por ejemplo tras un reintento parcial), se
    conserva la version mas reciente recibida, coherente con que la API puede
    revisar valores recientes.
    """

    base_url: str = _DEFAULT_BASE_URL
    series_title: str = "Demanda real"
    max_window_days: int = _DEFAULT_MAX_WINDOW_DAYS
    http_client: httpx.Client | None = None
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE

    @property
    def spec(self) -> SourceSpec:
        """Fuente de rol `TARGET`: demanda peninsular agregada, una unica serie."""
        return SourceSpec(
            source_id="ree_demand",
            role=Role.TARGET,
            value_columns=("y",),
            freq="h",
            native_tz=_NATIVE_TZ,
            vintage_aware=False,
            id_semantics="demanda electrica peninsular de Espana, en MW",
        )

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Descarga por tramos de fecha y devuelve el rango `[start, end)` en formato largo.

        Ver `chronolab.data.protocols.DataSource.fetch` para el contrato
        completo. `as_of` no esta soportado: esta fuente no es vintage-aware
        (la demanda en tiempo real no se revisa retroactivamente por vintage
        en este endpoint).
        """
        if as_of is not None:
            raise VintageNotSupported(
                f"{self.spec.source_id} no admite as_of (no es vintage-aware)"
            )

        client = self.http_client if self.http_client is not None else httpx.Client()
        chunks = _date_chunks(start, end, max_days=self.max_window_days)
        frames = [
            self._fetch_chunk(client, chunk_start, chunk_end) for chunk_start, chunk_end in chunks
        ]

        frame = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
        frame = deduplicate(frame, policy="last")
        frame = frame[(frame["ds"] >= start) & (frame["ds"] < end)]
        frame = frame.sort_values("ds").reset_index(drop=True)

        validated: pd.DataFrame = ree_demand_schema().validate(frame, lazy=True)
        return validated

    def _fetch_chunk(
        self, client: httpx.Client, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        start_local = start.tz_localize("UTC").tz_convert(_NATIVE_TZ)
        end_local = end.tz_localize("UTC").tz_convert(_NATIVE_TZ)
        params = {
            "start_date": start_local.strftime("%Y-%m-%dT%H:%M"),
            "end_date": end_local.strftime("%Y-%m-%dT%H:%M"),
            "time_trunc": "hour",
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
        return self._parse(response.json())

    def _parse(self, payload: Any) -> pd.DataFrame:
        values = None
        for series in payload.get("included", []):
            if series.get("attributes", {}).get("title") == self.series_title:
                values = series["attributes"]["values"]
                break
        if values is None:
            raise ValueError(
                f"{self.spec.source_id}: no se encontro la serie '{self.series_title}' "
                "en la respuesta de la API"
            )

        raw = pd.DataFrame(values)
        parsed_utc = pd.to_datetime(raw["datetime"], utc=True)
        return pd.DataFrame(
            {
                "unique_id": _UNIQUE_ID,
                "ds": to_utc_naive(parsed_utc),
                "y": raw["value"].astype("float64"),
            }
        )
