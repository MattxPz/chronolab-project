"""`CachedSource`: decorador de cache en parquet sobre cualquier `DataSource`.

La cache vive fuera de las fuentes para que cada fuente sea trivialmente
testeable y la politica de invalidacion se cambie en un unico lugar.
"""

import hashlib
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from chronolab.data.protocols import DataSource, SourceSpec
from chronolab.errors import SourceUnavailable, StaleCacheWarning
from chronolab.types import SeriesId

__all__ = ["CachedSource"]


@dataclass(frozen=True, slots=True)
class CachedSource:
    """Envuelve un `DataSource` con cache en parquet bajo `data/raw`.

    Implementa el protocolo `DataSource` por estructura: un `CachedSource` es a
    su vez una fuente valida, asi que se puede componer sin que el resto del
    codigo distinga una fuente cacheada de una que no lo esta.

    Parameters
    ----------
    inner
        Fuente real que se envuelve.
    cache_dir
        Raiz de la cache. Cada fuente escribe en su propio subdirectorio
        ``cache_dir / inner.spec.source_id``.
    max_age
        Antigüedad maxima que se sirve sin volver a consultar `inner`. Una
        entrada mas vieja que esto se considera invalida y se refresca.

    Notes
    -----
    La clave de cache se deriva de **todos** los parametros de la consulta
    (`source_id`, `start`, `end`, `ids`, `as_of`): dos llamadas con parametros
    distintos nunca comparten entrada, y la misma llamada repetida siempre
    resuelve al mismo fichero.

    Si `inner.fetch` lanza `SourceUnavailable` y existe una entrada de cache
    para esa clave, aunque este caducada, se sirve esa entrada con
    `StaleCacheWarning` en lugar de propagar el fallo: una respuesta obsoleta
    suele ser mas util que ninguna respuesta.
    """

    inner: DataSource
    cache_dir: Path = field(default_factory=lambda: Path("data/raw"))
    max_age: timedelta = field(default_factory=lambda: timedelta(days=1))

    @property
    def spec(self) -> SourceSpec:
        """Descripcion declarativa de la fuente envuelta."""
        return self.inner.spec

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Sirve desde cache si hay una entrada fresca; si no, consulta `inner`.

        Ver `chronolab.data.protocols.DataSource.fetch` para el contrato
        completo de parametros y de la trama devuelta.
        """
        path = self._cache_path(start=start, end=end, ids=ids, as_of=as_of)

        if path.is_file() and self._is_fresh(path):
            return pd.read_parquet(path)

        try:
            frame = self.inner.fetch(start=start, end=end, ids=ids, as_of=as_of)
        except SourceUnavailable:
            if path.is_file():
                warnings.warn(
                    f"{self.spec.source_id}: fuente no disponible, sirviendo "
                    f"cache obsoleta de {path.name}",
                    StaleCacheWarning,
                    stacklevel=2,
                )
                return pd.read_parquet(path)
            raise

        self._write(path, frame)
        return frame

    def _cache_key(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None,
        as_of: pd.Timestamp | None,
    ) -> str:
        payload = {
            "source_id": self.spec.source_id,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "ids": sorted(ids) if ids is not None else None,
            "as_of": as_of.isoformat() if as_of is not None else None,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def _cache_path(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None,
        as_of: pd.Timestamp | None,
    ) -> Path:
        key = self._cache_key(start=start, end=end, ids=ids, as_of=as_of)
        return self.cache_dir / self.spec.source_id / f"{key}.parquet"

    def _is_fresh(self, path: Path) -> bool:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return datetime.now(UTC) - modified <= self.max_age

    @staticmethod
    def _write(path: Path, frame: pd.DataFrame) -> None:
        """Escribe en parquet de forma atomica: fichero temporal + renombrado."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        frame.to_parquet(tmp_path)
        tmp_path.replace(path)
