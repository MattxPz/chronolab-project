"""Fuente UCI ElectricityLoadDiagrams 2011-2014: descarga y parseo.

Serie principal offline y siempre reproducible. Sin previsiones meteorologicas
archivadas para su rango, de ahi el vintage `SIMULATED_FORECAST` por defecto en
`chronolab.data.futr`.
"""

import io
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
import pandas as pd

from chronolab.data.align import deduplicate, resample_mean, to_utc_naive
from chronolab.data.protocols import SourceSpec
from chronolab.data.schemas import uci_electricity_schema
from chronolab.data.sources._http import (
    DEFAULT_BACKOFF_BASE,
    DEFAULT_MAX_RETRIES,
    request_with_retries,
)
from chronolab.errors import VintageNotSupported
from chronolab.types import Role, SeriesId

__all__ = ["UCIElectricitySource"]

_DEFAULT_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
_MEMBER_NAME = "LD2011_2014.txt"
_NATIVE_TZ = "Europe/Lisbon"
"""Zona horaria de las marcas de tiempo del fichero original, por ficha del dataset."""


@dataclass(frozen=True, slots=True)
class UCIElectricitySource:
    """`DataSource` para ElectricityLoadDiagrams 2011-2014 (UCI Machine Learning Repository).

    El fichero publicado es una trama **ancha**, separada por ``;``, con
    decimales en coma, muestreada cada 15 minutos, en hora local de Portugal
    (``Europe/Lisbon``), con una columna por cliente (``MT_001`` .. ``MT_370``)
    y valores de potencia en kW. Esta fuente descarga el zip, lo descomprime en
    memoria, pasa a formato largo, resuelve el cambio de hora con
    `chronolab.data.align.to_utc_naive`, deduplica y remuestrea a horario.

    Parameters
    ----------
    client_ids
        Subconjunto de clientes a devolver por defecto (por ejemplo
        ``("MT_001", "MT_042")``). ``None`` devuelve los 370. Se puede
        sobrescribir por llamada con el parametro `ids` de `fetch`. Filtrar
        aqui es lo que hace barato pedir solo unas pocas series del panel
        curado (docs/ARCHITECTURE.md D14) sin descartar trabajo: el filtrado
        ocurre antes de convertir a formato largo.
    url
        URL del zip. Parametrizable para tests y para copias alojadas
        localmente.
    http_client
        Cliente HTTP inyectable. Si es ``None``, se crea uno por defecto en
        cada llamada. En los tests se inyecta un cliente con
        ``transport=httpx.MockTransport(...)`` para no requerir red.
    timeout, max_retries, backoff_base
        Se reenvian a `request_with_retries`. El timeout por defecto es mayor
        que en las demas fuentes porque el zip pesa varios cientos de MB.

    Notes
    -----
    **Deduplicacion:** `policy="mean"`. En el vuelco de otono, la conversion a
    UTC ya distingue las dos ocurrencias de una hora local repetida como dos
    instantes UTC distintos (docstring de `to_utc_naive`), asi que en
    circunstancias normales no deberian quedar duplicados reales tras esa
    conversion. `"mean"` actua como politica de seguridad para cualquier
    duplicado residual —una descarga reintentada, una revision del fichero—
    sin preferir arbitrariamente una fila sobre otra.

    **Salto de primavera:** las horas locales que no existen ese dia se
    descartan (`to_utc_naive` las marca `NaT` y aqui se eliminan). El bucle
    horario resultante no tiene ninguna fila para ese instante; sera
    `reindex_to_full_grid`, aguas abajo en `assemble`, quien lo represente como
    un hueco explicito.
    """

    client_ids: tuple[str, ...] | None = None
    url: str = _DEFAULT_URL
    http_client: httpx.Client | None = None
    timeout: float = 30.0
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE

    @property
    def spec(self) -> SourceSpec:
        """Fuente de rol `TARGET`: cada cliente es una serie de consumo en kW."""
        return SourceSpec(
            source_id="uci_electricity",
            role=Role.TARGET,
            value_columns=("y",),
            freq="h",
            native_tz=_NATIVE_TZ,
            vintage_aware=False,
            id_semantics="cliente de consumo electrico (MT_001..MT_370)",
        )

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Descarga, parsea y devuelve el tramo `[start, end)` en formato largo.

        Ver `chronolab.data.protocols.DataSource.fetch` para el contrato
        completo. `as_of` no esta soportado: esta fuente no es vintage-aware.
        """
        if as_of is not None:
            raise VintageNotSupported(
                f"{self.spec.source_id} no admite as_of (no es vintage-aware)"
            )

        wide_local = self._parse(self._download())
        long_local = wide_local.melt(id_vars="ds_local", var_name="unique_id", value_name="y")

        selected: Sequence[str] | None = ids if ids is not None else self.client_ids
        if selected is not None:
            long_local = long_local[long_local["unique_id"].isin(selected)]

        long_local["ds"] = to_utc_naive(
            long_local["ds_local"], source_tz=_NATIVE_TZ, group=long_local["unique_id"]
        )
        long_local = long_local.dropna(subset=["ds"])

        frame = long_local[["unique_id", "ds", "y"]].astype({"y": "float64"})
        frame = deduplicate(frame, policy="mean")
        frame = resample_mean(frame, freq="h")
        frame = frame[(frame["ds"] >= start) & (frame["ds"] < end)]
        frame = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        validated: pd.DataFrame = uci_electricity_schema().validate(frame, lazy=True)
        return validated

    def _download(self) -> bytes:
        client = self.http_client if self.http_client is not None else httpx.Client()
        response = request_with_retries(
            client,
            "GET",
            self.url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
        )
        return response.content

    @staticmethod
    def _parse(raw_bytes: bytes) -> pd.DataFrame:
        """Descomprime el zip en memoria y devuelve la trama ancha con `ds_local`."""
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            member = _MEMBER_NAME if _MEMBER_NAME in archive.namelist() else archive.namelist()[0]
            with archive.open(member) as handle:
                wide = pd.read_csv(handle, sep=";", decimal=",", index_col=0)
        wide.index = pd.to_datetime(wide.index)
        return wide.reset_index(names="ds_local")
