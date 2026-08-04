"""Normalizacion temporal: UTC ingenuo, rejilla regular, huecos y duplicados.

Punto unico de conversion horaria del proyecto (`to_utc_naive`). Al vivir toda
la logica de DST aqui, el invariante I2 se audita en un solo sitio: cada fuente
llama a estas funciones y no reimplementa su propia conversion de huso horario.

El cambio de hora europeo produce dos anomalias en datos con marca de tiempo
*local*:

- **Salto de primavera** (ultimo domingo de marzo en Europa): el reloj local
  pasa de 02:00 a 03:00 de un salto. La hora 02:00-02:59 local **no existe**.
- **Vuelco de otono** (ultimo domingo de octubre): el reloj local retrocede de
  03:00 a 02:00. La hora 02:00-02:59 local **ocurre dos veces**: una en horario
  de verano (UTC+2) y otra en horario de invierno (UTC+1).

En UTC ninguna de las dos anomalias existe: UTC no observa cambio de hora, asi
que convertir a UTC ingenuo en el borde de la fuente, y trabajar solo en UTC a
partir de ahi, es lo que elimina el problema por construccion en lugar de
parchearlo despues.
"""

from typing import Literal

import pandas as pd

__all__ = ["DedupPolicy", "deduplicate", "reindex_to_full_grid", "resample_mean", "to_utc_naive"]

DedupPolicy = Literal["first", "last", "mean"]
"""Politica de deduplicacion de timestamps repetidos. Ver `deduplicate`."""


def to_utc_naive(
    ds: pd.Series,
    *,
    source_tz: str = "UTC",
    group: pd.Series | None = None,
) -> pd.Series:
    """Convierte una columna de tiempo a UTC ingenuo.

    Acepta dos formas de entrada:

    1. **Tz-aware**, con el huso horario ya incluido en cada marca (por ejemplo
       ISO 8601 con offset explicito, ``"2024-10-27T02:30:00+02:00"``). Este
       caso no tiene ambiguedad posible: cada timestamp ya sabe a que instante
       UTC corresponde. Se convierte y se retira el huso.
    2. **Naive**, representando hora de pared local en `source_tz`. Este es el
       caso que requiere resolver DST.

    Parameters
    ----------
    ds
        Serie de timestamps (``datetime64[ns]``, con o sin huso).
    source_tz
        Zona horaria IANA en la que `ds` esta expresada, cuando `ds` es naive.
        Si `ds` ya es tz-aware, se ignora. Si es ``"UTC"`` y `ds` es naive, se
        devuelve tal cual: ya esta en la representacion canonica.
    group
        Clave de agrupacion opcional, de la misma longitud que `ds` (tipicamente
        ``frame["unique_id"]``). Es **obligatoria** cuando `ds` puede contener
        el mismo timestamp local repetido por un motivo *distinto* de la
        ambiguedad de DST, por ejemplo varias series compartiendo el mismo
        indice temporal. Sin `group`, ese caso se confundiria con una hora
        duplicada por vuelco de otono. Si `ds` cubre una unica serie, se puede
        omitir.

    Returns
    -------
    pandas.Series
        Timestamps en UTC ingenuo (``datetime64[ns]``, sin huso). Las marcas
        cuya hora local *no existe* (salto de primavera) se devuelven como
        ``NaT``: son un artefacto del reloj de pared, no una observacion real,
        y el llamante debe descartarlas antes de continuar.

    Notes
    -----
    Para resolver la ambiguedad del vuelco de otono se asume que `ds` (y
    `group`, si se pasa) llegan en **orden cronologico de aparicion**: dentro de
    cada grupo, la primera vez que se ve un timestamp local repetido es la
    ocurrencia en horario de verano (UTC+2 en Europa), y la segunda es la
    ocurrencia en horario de invierno (UTC+1), que es exactamente el orden en
    que el reloj de pared las produce. Todas las fuentes de este proyecto
    entregan sus datos ya ordenados por tiempo, asi que esta condicion se
    cumple sin trabajo adicional.

    Examples
    --------
    Vuelco de otono de 2024 en Europe/Madrid: la 01:30 UTC ocurre una vez, pero
    corresponde a dos horas de pared locales distintas (03:30 y 02:30 CET/CEST
    respectivamente) solo si tomamos horas de pared reales; aqui se ilustra el
    caso simetrico, dos apariciones de la misma hora de pared que resuelven a
    instantes UTC distintos:

    >>> local = pd.Series(pd.to_datetime(["2024-10-27 02:30", "2024-10-27 02:30"]))
    >>> to_utc_naive(local, source_tz="Europe/Madrid").tolist()
    [Timestamp('2024-10-27 00:30:00'), Timestamp('2024-10-27 01:30:00')]
    """
    if ds.dt.tz is not None:
        converted: pd.Series = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        return converted

    if source_tz == "UTC":
        return ds

    key = ds.astype(str) if group is None else group.astype(str) + "|" + ds.astype(str)
    occurrence = key.groupby(key).cumcount()
    is_dst = (occurrence == 0).to_numpy()

    localized = ds.dt.tz_localize(source_tz, ambiguous=is_dst, nonexistent="NaT")
    result: pd.Series = localized.dt.tz_convert("UTC").dt.tz_localize(None)
    return result


def reindex_to_full_grid(frame: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    """Completa la rejilla temporal de cada serie, con `NaN` explicito en los huecos.

    Es la implementacion del invariante I3: "falta un dato" debe ser una fila
    con valor nulo, nunca una fila ausente. Sin esto, un modelo estacional que
    asume rejilla regular se desalinea en silencio en cuanto hay un hueco.

    Parameters
    ----------
    frame
        Trama larga con columnas ``unique_id``, ``ds`` y columnas de valor.
        `ds` debe estar ya en UTC ingenuo (salida de `to_utc_naive`) y sin
        duplicados dentro de cada serie (salida de `deduplicate`).
    freq
        Alias de offset de pandas de la rejilla, por ejemplo ``"h"``.

    Returns
    -------
    pandas.DataFrame
        Misma columnas que `frame`, ordenada por ``(unique_id, ds)``. Cada
        serie cubre exactamente ``[min(ds), max(ds)]`` a la frecuencia `freq`;
        las marcas que faltaban en la entrada aparecen con las columnas de
        valor a `NaN`. El rango de cada serie lo decide la propia serie: esta
        funcion no extiende una serie mas alla de su primera o ultima
        observacion real.
    """
    value_columns = [c for c in frame.columns if c not in ("unique_id", "ds")]
    parts: list[pd.DataFrame] = []
    for uid, group in frame.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds")
        full_index = pd.date_range(ordered["ds"].min(), ordered["ds"].max(), freq=freq)
        reindexed = ordered.set_index("ds")[value_columns].reindex(full_index)
        reindexed.index.name = "ds"
        reindexed.insert(0, "unique_id", uid)
        parts.append(reindexed.reset_index())

    result = pd.concat(parts, ignore_index=True)
    return result[["unique_id", "ds", *value_columns]]


def deduplicate(frame: pd.DataFrame, *, policy: DedupPolicy = "mean") -> pd.DataFrame:
    """Elimina timestamps repetidos dentro de cada serie segun una politica declarada.

    Un duplicado es un par ``(unique_id, ds)`` que aparece mas de una vez. En la
    practica esto ocurre por reintentos de una API que reenvian el mismo tramo,
    por una revision de datos que anade una fila nueva sin retirar la vieja, o
    por un vuelco de otono mal resuelto aguas arriba.

    Parameters
    ----------
    frame
        Trama larga con columnas ``unique_id``, ``ds`` y columnas de valor.
    policy
        - ``"mean"`` (por defecto): promedia las columnas de valor de todas las
          filas duplicadas. Es la eleccion por defecto porque, sin una razon
          para preferir una fila sobre otra, promediar es la que menos sesga.
        - ``"first"``: conserva la primera fila tal como llega y descarta el
          resto. Apropiado cuando el orden de llegada no aporta informacion
          adicional y se prefiere el dato mas antiguo.
        - ``"last"``: conserva la ultima fila. Apropiado cuando el orden de
          llegada codifica una revision, es decir, la fila mas reciente es la
          mas fiable.

    Returns
    -------
    pandas.DataFrame
        Sin duplicados por ``(unique_id, ds)``. Con `policy="mean"` el orden de
        columnas y filas queda determinado por el `groupby`; con `"first"` o
        `"last"` se conserva el orden original de `frame`.
    """
    value_columns = [c for c in frame.columns if c not in ("unique_id", "ds")]

    if policy == "mean":
        grouped = frame.groupby(["unique_id", "ds"], as_index=False, sort=False)[value_columns]
        return grouped.mean()[["unique_id", "ds", *value_columns]]

    keep: Literal["first", "last"] = policy
    return frame.drop_duplicates(subset=["unique_id", "ds"], keep=keep).reset_index(drop=True)


def resample_mean(frame: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    """Remuestrea cada serie a una frecuencia mas gruesa promediando cada bucket.

    Se usa para llevar UCI de 15 minutos a horario. La media es la agregacion
    correcta para una magnitud de potencia (kW): el promedio de las lecturas de
    potencia dentro de una hora coincide numericamente con la energia
    consumida en esa hora expresada en kWh. Si la magnitud de origen fuese ya
    energia acumulada por intervalo, sumar seria lo correcto; esta funcion no
    lo decide por si sola, y el llamante debe elegir la agregacion adecuada a
    sus unidades antes de usarla si esas unidades difieren.

    Parameters
    ----------
    frame
        Trama larga con columnas ``unique_id``, ``ds`` y columnas de valor.
        `ds` debe estar en UTC ingenuo.
    freq
        Frecuencia destino, mas gruesa que la de origen, por ejemplo ``"h"``.

    Returns
    -------
    pandas.DataFrame
        Misma columnas que `frame`, con una fila por ``(unique_id, bucket)``.
        Un bucket sin ninguna observacion de origen se conserva con `NaN`, en
        lugar de desaparecer, para no reintroducir el problema que resuelve
        `reindex_to_full_grid`.
    """
    value_columns = [c for c in frame.columns if c not in ("unique_id", "ds")]
    parts: list[pd.DataFrame] = []
    for uid, group in frame.groupby("unique_id", sort=False):
        resampled = group.set_index("ds")[value_columns].resample(freq).mean()
        resampled.insert(0, "unique_id", uid)
        parts.append(resampled.reset_index())

    result = pd.concat(parts, ignore_index=True)
    return result[["unique_id", "ds", *value_columns]]
