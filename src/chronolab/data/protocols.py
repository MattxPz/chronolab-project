"""Protocolo `DataSource` y su especificacion declarativa.

Una fuente **no** limpia, **no** completa huecos y **no** valida invariantes de
panel: entrega lo que tiene, en formato largo y en UTC ingenuo. La limpieza vive
en `chronolab.data.align` y el ensamblado en `chronolab.data.assemble`. La
separacion es deliberada: si cada fuente limpiase a su manera, no habria un solo
sitio donde auditar el tratamiento de huecos y de DST.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from chronolab.types import Role, SeriesId

__all__ = ["DataSource", "SourceSpec"]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Descripcion declarativa de lo que produce una fuente.

    Parameters
    ----------
    source_id
        Identificador estable; forma parte de la clave de cache.
    role
        Papel semantico de las columnas que entrega. Una fuente tiene un unico
        rol; una fuente que produjera columnas de roles distintos se parte en dos.
        Forzar esto evita la abstraccion que se filtra: obligar a Open-Meteo a
        devolver una columna ``y`` que luego se renombra es la clase de comodidad
        que despues nadie sabe interpretar.
    value_columns
        Nombres de las columnas de valor que devuelve `fetch`, sin las claves.
    freq
        Frecuencia nativa de la fuente. Si difiere de la del panel, `assemble`
        remuestrea con una agregacion declarada explicitamente.
    native_tz
        Zona horaria en la que la fuente publica sus marcas de tiempo. Se usa una
        sola vez, en la conversion a UTC ingenuo.
    vintage_aware
        ``True`` si la fuente sabe responder "que se sabia en `as_of`". Si es
        ``False``, pasar `as_of` es un error, no un parametro que se ignora.
    id_semantics
        Que representa `unique_id` en esta fuente: cliente, zona de mercado,
        simbolo. Va al model card del dataset.
    """

    source_id: str
    role: Role
    value_columns: tuple[str, ...]
    freq: str
    native_tz: str = "UTC"
    vintage_aware: bool = False
    id_semantics: str = ""


@runtime_checkable
class DataSource(Protocol):
    """Contrato de obtencion de datos crudos."""

    @property
    def spec(self) -> SourceSpec:
        """Descripcion declarativa de la fuente. Constante durante su vida."""
        ...

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Obtiene datos crudos en formato largo.

        Parameters
        ----------
        start, end
            Intervalo **semiabierto** ``[start, end)`` en UTC ingenuo. La
            semiapertura es obligatoria y uniforme en todo el proyecto: los
            intervalos cerrados son la causa clasica del solape de un punto entre
            entrenamiento y evaluacion.
        ids
            Series a obtener. ``None`` significa "todas las que ofrezca la
            fuente". Son los identificadores de la fuente; el renombrado canonico
            ocurre en `assemble`.
        as_of
            Instante de conocimiento: devuelve la informacion tal y como estaba
            publicada en ese momento. Solo admisible si ``spec.vintage_aware``.

        Returns
        -------
        pandas.DataFrame
            Columnas ``unique_id``, ``ds`` (datetime64[ns], UTC ingenuo) y
            ``spec.value_columns``. Puede tener huecos, puede no estar ordenado y
            puede traer duplicados: eso es problema de `align`, no del llamante.

        Raises
        ------
        VintageNotSupported
            Si se pasa `as_of` a una fuente con ``vintage_aware=False``.
        SourceUnavailable
            Si la fuente remota no responde. `CachedSource` la captura y sirve la
            ultima version valida marcandola como obsoleta.
        """
        ...
