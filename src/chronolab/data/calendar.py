"""Festivos, terminos de Fourier y features de calendario local seguras ante DST.

Unico modulo donde conviven UTC y hora local: "es festivo" u "hora del dia" son
propiedades del tiempo local, pero las columnas que produce este modulo se
devuelven alineadas al indice UTC del panel, es decir, con la misma longitud y
el mismo orden que la columna `ds` de entrada.

Todas las features de este modulo son funcion exclusivamente de la marca de
tiempo: no dependen de ningun valor observado. En terminos de
`chronolab.features.roles`, su `max_lead` es infinito.
"""

from collections.abc import Mapping

import holidays as holidays_lib
import numpy as np
import pandas as pd

__all__ = [
    "calendar_features",
    "fourier_terms",
    "holiday_eve_flags",
    "holiday_flags",
    "local_hour",
]

_EPOCH = pd.Timestamp("1970-01-01")


def _to_local(ds: pd.Series, *, tz_display: str) -> pd.DatetimeIndex:
    """Reexpresa una columna `ds` en UTC ingenuo como hora local de lectura.

    Uso interno: el resultado sirve para calcular hora del dia, dia de la
    semana o festivo segun el calendario local, y nunca se devuelve tal cual.
    """
    index = pd.DatetimeIndex(ds)
    if index.tz is not None:
        raise ValueError("ds debe estar en UTC ingenuo (sin huso horario)")
    return index.tz_localize("UTC").tz_convert(tz_display)


def local_hour(ds: pd.Series, *, tz_display: str) -> pd.Series:
    """Hora del dia local (0-23) de cada marca de tiempo.

    Es la version publica y minima de la conversion que ya hacia
    `calendar_features`: hay consumidores —la calibracion de Mondrian del
    detector conformal— que necesitan la hora local y **solo** la hora local, y
    que no deben rederivarla por su cuenta. Este modulo es el unico sitio del
    proyecto donde conviven UTC y hora local, y esa propiedad se mantiene
    exportando la conversion en lugar de duplicandola.

    La direccion UTC -> local es total y no ambigua, tambien en los cambios de
    hora: la ambigua es la inversa, que es justamente por lo que el invariante
    I2 almacena UTC ingenuo.

    Parameters
    ----------
    ds
        Columna de tiempo en UTC ingenuo.
    tz_display
        Zona horaria de lectura, normalmente ``spec.tz_display``.

    Returns
    -------
    pandas.Series
        Enteros ``int16`` en ``[0, 23]``, con el mismo indice, longitud y orden
        que `ds`.

    Examples
    --------
    >>> ds = pd.Series(pd.to_datetime(["2023-06-01 00:00", "2023-06-01 12:00"]))
    >>> local_hour(ds, tz_display="Europe/Madrid").tolist()
    [2, 14]
    """
    local = _to_local(ds, tz_display=tz_display)
    return pd.Series(local.hour.to_numpy().astype(np.int16), index=ds.index, name="hour")


def holiday_flags(
    ds: pd.Series,
    *,
    country: str,
    tz_display: str = "UTC",
    subdiv: str | None = None,
) -> pd.Series:
    """Marca si cada marca de tiempo cae en un dia festivo del calendario local.

    Parameters
    ----------
    ds
        Columna de tiempo en UTC ingenuo.
    country
        Codigo ISO 3166-1 alpha-2 del pais, por ejemplo ``"ES"`` o ``"PT"``.
    tz_display
        Zona horaria local con la que se determina la fecha civil: un instante
        UTC puede caer en un dia distinto segun el huso de lectura, y el
        festivo se decide por la fecha local, no por la fecha UTC.
    subdiv
        Subdivision opcional del pais (por ejemplo una comunidad autonoma),
        para calendarios con festivos regionales.

    Returns
    -------
    pandas.Series
        Booleana, con el mismo indice que `ds` y nombre ``"is_holiday"``.
    """
    local = _to_local(ds, tz_display=tz_display)
    years = sorted({timestamp.year for timestamp in local})
    calendar = holidays_lib.country_holidays(country, subdiv=subdiv, years=years)
    is_holiday = np.fromiter(
        (timestamp.date() in calendar for timestamp in local), dtype=bool, count=len(local)
    )
    return pd.Series(is_holiday, index=ds.index, name="is_holiday")


def holiday_eve_flags(
    ds: pd.Series,
    *,
    country: str,
    tz_display: str = "UTC",
    subdiv: str | None = None,
) -> pd.Series:
    """Marca si el dia civil **siguiente** a cada marca de tiempo es festivo.

    La vispera de un festivo es una feature con valor propio en demanda
    electrica: el patron de consumo de un domingo que precede a un lunes
    festivo se parece mas al de un festivo que al de un domingo cualquiera.
    Como `holiday_flags`, la fecha civil se decide por `tz_display`, no por UTC.

    Parameters
    ----------
    ds
        Columna de tiempo en UTC ingenuo.
    country
        Codigo ISO 3166-1 alpha-2 del pais.
    tz_display
        Zona horaria local con la que se determina la fecha civil.
    subdiv
        Subdivision opcional del pais, para calendarios con festivos
        regionales.

    Returns
    -------
    pandas.Series
        Booleana, con el mismo indice que `ds` y nombre ``"is_holiday_eve"``.
    """
    local = _to_local(ds, tz_display=tz_display)
    # +1 al final del rango de anos: el 31 de diciembre mira al 1 de enero
    # del ano siguiente, que si no se incluye aqui el calendario no conoce.
    years = sorted({timestamp.year for timestamp in local} | {local.max().year + 1})
    calendar = holidays_lib.country_holidays(country, subdiv=subdiv, years=years)
    next_day = (local + pd.Timedelta(days=1)).date
    is_eve = np.fromiter((day in calendar for day in next_day), dtype=bool, count=len(local))
    return pd.Series(is_eve, index=ds.index, name="is_holiday_eve")


def calendar_features(
    ds: pd.Series,
    *,
    tz_display: str = "UTC",
    country: str | None = None,
    subdiv: str | None = None,
) -> pd.DataFrame:
    """Genera las features de calendario estandar del proyecto.

    Parameters
    ----------
    ds
        Columna de tiempo en UTC ingenuo.
    tz_display
        Zona horaria local desde la que se leen hora del dia, dia de la semana
        y mes. Sin esto, "es festivo" o "es la hora punta" quedarian atados al
        reloj UTC, que no es el que vive nadie.
    country
        Si se indica, anade las columnas ``is_holiday`` (via `holiday_flags`) y
        ``is_holiday_eve`` (via `holiday_eve_flags`). Si es ``None``, ninguna de
        las dos se genera.
    subdiv
        Se reenvia a `holiday_flags` y `holiday_eve_flags` cuando `country`
        esta presente.

    Returns
    -------
    pandas.DataFrame
        Con el mismo indice que `ds` y las columnas: ``ds`` (la entrada, sin
        modificar), ``hour`` (0-23), ``dayofweek`` (0=lunes .. 6=domingo),
        ``day`` (1-31, dia del mes), ``month`` (1-12), ``is_weekend``, los
        pares seno/coseno de hora, dia de la semana y mes (``hour_sin``,
        ``hour_cos``, ``dow_sin``, ``dow_cos``, ``month_sin``, ``month_cos``) y,
        si `country` no es ``None``, ``is_holiday`` e ``is_holiday_eve``.
    """
    local = _to_local(ds, tz_display=tz_display)
    hour = local.hour.to_numpy().astype(np.float64)
    dow = local.dayofweek.to_numpy().astype(np.float64)
    day = local.day.to_numpy().astype(np.float64)
    month = local.month.to_numpy().astype(np.float64)

    frame = pd.DataFrame(
        {
            "ds": ds.to_numpy(),
            "hour": hour.astype(np.int16),
            "dayofweek": dow.astype(np.int16),
            "day": day.astype(np.int16),
            "month": month.astype(np.int16),
            "is_weekend": dow >= 5,
            "hour_sin": np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0).astype(np.float32),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0).astype(np.float32),
            "month_sin": np.sin(2 * np.pi * (month - 1.0) / 12.0).astype(np.float32),
            "month_cos": np.cos(2 * np.pi * (month - 1.0) / 12.0).astype(np.float32),
        },
        index=ds.index,
    )

    if country is not None:
        frame["is_holiday"] = holiday_flags(
            ds, country=country, tz_display=tz_display, subdiv=subdiv
        ).to_numpy()
        frame["is_holiday_eve"] = holiday_eve_flags(
            ds, country=country, tz_display=tz_display, subdiv=subdiv
        ).to_numpy()

    return frame


def fourier_terms(ds: pd.Series, *, periods: Mapping[str, float], order: int = 2) -> pd.DataFrame:
    """Terminos de Fourier de una o mas estacionalidades, como regresoras deterministas.

    Alternativa continua a las variables indicadoras de calendario: en vez de
    una dummy por hora del dia, un par seno/coseno por armonico aproxima la
    misma estacionalidad con muchas menos columnas y sin discontinuidad en la
    frontera del ciclo.

    Parameters
    ----------
    ds
        Columna de tiempo en UTC ingenuo.
    periods
        Mapa ``{nombre: longitud_en_horas}`` de cada estacionalidad, por
        ejemplo ``{"daily": 24.0, "weekly": 168.0}``. Se expresa en horas y no
        en pasos de rejilla porque el calculo usa el tiempo transcurrido real
        desde una referencia fija: a diferencia de una version basada en la
        posicion de la fila, esta sigue siendo correcta aunque `ds` tenga
        huecos.
    order
        Numero de armonicos por estacionalidad.

    Returns
    -------
    pandas.DataFrame
        Con el mismo indice que `ds` y columnas ``ds`` mas
        ``fourier_{nombre}_sin_{k}`` / ``fourier_{nombre}_cos_{k}`` para
        ``k`` en ``1..order``.
    """
    hours_elapsed = (pd.DatetimeIndex(ds) - _EPOCH) / pd.Timedelta(hours=1)
    frame = pd.DataFrame({"ds": ds.to_numpy()}, index=ds.index)
    for name, period_hours in periods.items():
        for k in range(1, order + 1):
            angle = 2 * np.pi * k * hours_elapsed / period_hours
            frame[f"fourier_{name}_sin_{k}"] = np.sin(angle).astype(np.float32)
            frame[f"fourier_{name}_cos_{k}"] = np.cos(angle).astype(np.float32)
    return frame
