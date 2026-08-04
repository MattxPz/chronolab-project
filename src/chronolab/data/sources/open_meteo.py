"""Fuente Open-Meteo: temperatura realizada (archive) y prevista (historical-forecast).

Ambos endpoints son fuentes distintas con el mismo cliente: el archivo es
reanalisis (valor revisado) y no es lo que se sabia en el cutoff. Este modulo
implementa por ahora solo el endpoint de archivo; `historical-forecast`
(vintage `ARCHIVED_FORECAST`) queda pendiente de una clase hermana.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from chronolab.data.align import deduplicate, to_utc_naive
from chronolab.data.protocols import SourceSpec
from chronolab.data.schemas import open_meteo_schema
from chronolab.data.sources._http import (
    DEFAULT_BACKOFF_BASE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    QueryValue,
    request_with_retries,
)
from chronolab.errors import VintageNotSupported
from chronolab.types import Role, SeriesId

__all__ = ["OpenMeteoSource"]

_DEFAULT_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


@dataclass(frozen=True, slots=True)
class OpenMeteoSource:
    """`DataSource` para temperatura horaria de Open-Meteo (endpoint de archivo).

    Envuelve ``/v1/archive``, que sirve **reanalisis**: el valor de temperatura
    revisado a posteriori para una coordenada y un rango de fechas, no una
    prevision. docs/ARCHITECTURE.md §4.3 es explicito sobre esta distincion:
    usar este valor como si fuese lo que se sabia en el cutoff es la fuga de
    informacion con mas capacidad de invalidar el resultado principal del
    proyecto, precisamente porque no deja ningun sintoma visible. Esta fuente
    declara ``vintage_aware=False`` para que quien la use no pueda pasar
    `as_of` y olvidar la distincion: el `FutrProvider` que la envuelva debe
    declarar explicitamente el vintage que corresponde (`SIMULATED_FORECAST`
    casi siempre; `ARCHIVED_FORECAST` exigiria el endpoint separado
    ``historical-forecast``, que esta clase no implementa).

    La consulta se pide con ``timezone=UTC``: Open-Meteo devuelve entonces
    ``hourly.time`` ya en UTC ingenuo, sin desplazamiento horario que
    resolver. Es la eleccion que evita el problema de DST en el origen, en
    lugar de resolverlo despues con `to_utc_naive`.

    Parameters
    ----------
    latitude, longitude
        Coordenada de la estacion virtual.
    location_id
        Identificador de la serie. Si es ``None``, se deriva de la coordenada
        como ``f"{latitude:.4f}_{longitude:.4f}"``.
    base_url
        URL del endpoint. Parametrizable para tests.
    http_client, timeout, max_retries, backoff_base
        Ver `chronolab.data.sources._http.request_with_retries`.

    Notes
    -----
    Deduplicacion: ``policy="last"``. Open-Meteo no deberia repetir marcas de
    tiempo dentro de una misma respuesta; la politica solo protege ante un
    reintento que reenvie un tramo ya recibido.
    """

    latitude: float
    longitude: float
    location_id: str | None = None
    base_url: str = _DEFAULT_BASE_URL
    http_client: httpx.Client | None = None
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE

    @property
    def spec(self) -> SourceSpec:
        """Fuente de rol `FUTR_EXOG`: temperatura, pensada como exogena futura."""
        return SourceSpec(
            source_id="open_meteo_archive",
            role=Role.FUTR_EXOG,
            value_columns=("temp_c",),
            freq="h",
            native_tz="UTC",
            vintage_aware=False,
            id_semantics="coordenada geografica (latitud_longitud)",
        )

    @property
    def _location_id(self) -> str:
        if self.location_id is not None:
            return self.location_id
        return f"{self.latitude:.4f}_{self.longitude:.4f}"

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Descarga temperatura horaria realizada para `[start, end)`.

        Ver `chronolab.data.protocols.DataSource.fetch`. `as_of` no esta
        soportado: el endpoint de archivo sirve reanalisis, no versiones por
        fecha de consulta.
        """
        if as_of is not None:
            raise VintageNotSupported(
                f"{self.spec.source_id} no admite as_of (no es vintage-aware)"
            )

        client = self.http_client if self.http_client is not None else httpx.Client()
        params: dict[str, QueryValue] = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start.strftime("%Y-%m-%d"),
            # `end` es exclusivo (semiapertura obligatoria en todo el
            # proyecto); Open-Meteo solo entiende fechas inclusivas, asi que
            # se retrocede un segundo antes de tomar la fecha del ultimo dia.
            "end_date": (end - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d"),
            "hourly": "temperature_2m",
            "timezone": "UTC",
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
        frame = self._parse(response.json())
        frame = deduplicate(frame, policy="last")
        frame = frame[(frame["ds"] >= start) & (frame["ds"] < end)]
        frame = frame.sort_values("ds").reset_index(drop=True)

        validated: pd.DataFrame = open_meteo_schema().validate(frame, lazy=True)
        return validated

    def _parse(self, payload: Any) -> pd.DataFrame:
        hourly = payload["hourly"]
        ds_local = pd.to_datetime(pd.Series(hourly["time"]))
        return pd.DataFrame(
            {
                "unique_id": self._location_id,
                "ds": to_utc_naive(ds_local, source_tz="UTC"),
                "temp_c": pd.Series(hourly["temperature_2m"], dtype="float64"),
            }
        )
