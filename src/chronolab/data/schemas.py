"""Esquemas pandera de las tramas crudas y del panel canonico.

Codifica los invariantes I1-I7 de docs/ARCHITECTURE.md §3.3, que se comprueban
una sola vez en `assemble.build_panel` y a partir de ahi se asumen. Antes de eso
esta el esquema de este modulo para las tramas *crudas* que entrega cada
`DataSource`: no exige rejilla completa ni ausencia de huecos (de eso se ocupa
`align`), pero si exige tipos correctos, monotonia del indice temporal *dentro
de cada serie*, ausencia de duplicados exactos y valores dentro de un rango
plausible. Sin esta validacion, un fallo de parseo aguas arriba (una columna
que llega como texto, una fecha invertida, un sensor que reporta -9999) se
propaga en silencio hasta el modelo.

La validacion corre al salir de cada fuente: cada `fetch()` termina con
``schema.validate(frame, lazy=True)`` antes de devolver.
"""

from collections.abc import Mapping

import pandas as pd
import pandera.pandas as pa

__all__ = [
    "binance_schema",
    "build_raw_schema",
    "open_meteo_schema",
    "ree_demand_schema",
    "uci_electricity_schema",
]


def _is_monotonic_per_series(frame: pd.DataFrame) -> bool:
    """Verifica que ``ds`` sea estrictamente creciente dentro de cada ``unique_id``.

    Monotonia *por serie* y no global: dos series distintas pueden compartir
    marcas de tiempo sin que eso sea un problema, asi que una comprobacion
    global rechazaria paneles multi-serie perfectamente validos.

    Un DataFrame vacio (una consulta valida que no devolvio filas) es
    vacuamente monotono: no hay ningun par de filas que lo contradiga. Sin
    este caso base, `groupby(...).apply(...)` sobre cero grupos deja a pandas
    sin forma de inferir el tipo del resultado y `.all()` revienta con un
    `TypeError` ajeno al problema real.
    """
    if frame.empty:
        return True
    return bool(
        frame.groupby("unique_id")["ds"]
        .apply(lambda s: s.is_monotonic_increasing and s.is_unique)
        .all()
    )


def build_raw_schema(
    value_columns: Mapping[str, tuple[float, float]],
    *,
    coerce: bool = True,
) -> pa.DataFrameSchema:
    """Construye el esquema de una trama cruda en formato largo.

    Parameters
    ----------
    value_columns
        Mapa ``{nombre_columna: (minimo, maximo)}`` con el rango plausible de
        cada columna de valor. El rango es una comprobacion de cordura, no una
        garantia fisica: existe para atrapar errores de unidad o de parseo
        (grados Fahrenheit donde se esperaban Celsius, kW donde se esperaba MW),
        no para rechazar valores extremos pero reales.
    coerce
        Si es ``True``, pandera intenta convertir los tipos antes de validar.
        Se activa por defecto porque las fuentes externas suelen entregar
        numeros como texto o como `object`.

    Returns
    -------
    pandera.DataFrameSchema
        Esquema con columnas ``unique_id`` (str), ``ds`` (datetime64[ns], sin
        huso) y una columna `float64` por entrada de `value_columns`, mas la
        comprobacion de monotonia por serie a nivel de trama completa.

    Notes
    -----
    Deliberadamente **no** se exige aqui que ``ds`` este en UTC ingenuo: esa
    conversion es responsabilidad de ``align.to_utc_naive`` y ocurre *dentro*
    de cada fuente, antes de que la trama llegue a este esquema. Cuando este
    esquema se aplica, ``ds`` ya deberia ser tz-naive por construccion; si no lo
    es, `pandera` lo rechaza por el tipo declarado.
    """
    columns: dict[str, pa.Column] = {
        "unique_id": pa.Column(str, nullable=False),
        "ds": pa.Column("datetime64[ns]", nullable=False),
    }
    for name, (lo, hi) in value_columns.items():
        columns[name] = pa.Column(
            float,
            checks=pa.Check.in_range(lo, hi, include_min=True, include_max=True),
            nullable=True,  # un hueco de origen es un NaN legitimo, no un error
        )

    return pa.DataFrameSchema(
        columns,
        checks=pa.Check(
            _is_monotonic_per_series,
            error="ds debe ser estrictamente creciente y sin duplicados dentro de cada unique_id",
        ),
        coerce=coerce,
        strict=True,
    )


def uci_electricity_schema() -> pa.DataFrameSchema:
    """Esquema de `UCIElectricitySource`: consumo horario en kW por cliente."""
    return build_raw_schema({"y": (0.0, 50_000.0)})


def ree_demand_schema() -> pa.DataFrameSchema:
    """Esquema de `REEDemandSource`: demanda electrica horaria de Espana en MW."""
    return build_raw_schema({"y": (0.0, 60_000.0)})


def open_meteo_schema() -> pa.DataFrameSchema:
    """Esquema de `OpenMeteoSource`: temperatura horaria en grados Celsius.

    El rango ``[-40, 55]`` cubre con margen los extremos registrados en
    superficie; sirve para atrapar errores de unidad, no para modelar el clima.
    """
    return build_raw_schema({"temp_c": (-40.0, 55.0)})


def binance_schema() -> pa.DataFrameSchema:
    """Esquema de `BinanceKlinesSource`: precio de cierre por vela horaria."""
    return build_raw_schema({"y": (0.0, 10_000_000.0)})
