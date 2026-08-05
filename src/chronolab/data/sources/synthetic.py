"""Generador determinista de series sinteticas para tests y modo demo sin red.

Permite construir paneles con estacionalidad conocida, huecos y anomalias
controladas, que es lo que hace verificable el arnes de evaluacion.

**Estas series no son datos reales.** Son un sustituto deliberadamente
etiquetado para dos usos: (1) tests que necesitan datos con estructura
conocida, y (2) el modo demo de la EDA y de la app cuando no hay red para
descargar UCI, REE u Open-Meteo. Cualquier hallazgo derivado de estas series
describe el pipeline, no el mundo.

`SyntheticElectricitySource` (rol `TARGET`) genera tres series con una
dificultad de prediccion deliberadamente escalonada:

- ``residential_north``: estacionalidad diaria y semanal fuerte, tendencia
  suave, ruido bajo, respuesta termica marcada. Deberia ser la mas facil.
- ``commercial_mixed``: estacionalidad de horario comercial (fuerte entre
  semana, casi nula el fin de semana), un par de escalones de nivel, ruido
  moderado. Dificultad media.
- ``volatile_industrial``: perfil casi plano con eventos de lote irregulares
  (llegadas de Poisson) y ruido alto. Debilmente estacional y dificil de
  prever: cumple el mismo papel pedagogico que la serie de contraste cripto
  de docs/PLAN_PROYECTO.md §0, sin depender de una API externa.

`SyntheticWeatherSource` (rol `FUTR_EXOG`) genera la temperatura horaria que
alimenta la respuesta termica de las tres series de demanda, con la misma
semilla: ambas fuentes son mutuamente consistentes sin depender la una de la
otra.

Ambas fuentes generan siempre su ventana completa fija internamente y filtran
al final al rango pedido: asi, pedir sub-rangos distintos del mismo periodo da
siempre los mismos valores en la parte que se solapa, que es lo que se espera
de una fuente real.
"""

from collections.abc import Sequence

import holidays as holidays_lib
import numpy as np
import pandas as pd

from chronolab.data.protocols import SourceSpec
from chronolab.errors import VintageNotSupported
from chronolab.types import Role, SeriesId

__all__ = ["DEMO_SERIES_IDS", "SyntheticElectricitySource", "SyntheticWeatherSource"]

DEMO_SERIES_IDS: tuple[str, str, str] = (
    "residential_north",
    "commercial_mixed",
    "volatile_industrial",
)
"""Identificadores de las tres series de demanda sinteticas."""

_NATIVE_TZ = "Europe/Madrid"

# Ventana fija interna: 14 meses horarios, suficiente para MSTL(24,168),
# ACF/PACF a 200 retardos y un periodograma con resolucion util, y que cubre
# dos transiciones de DST reales: el vuelco de otono de 2023-10-29 y el salto
# de primavera de 2024-03-31.
_FULL_START = pd.Timestamp("2023-06-01", tz="UTC")
_FULL_END = pd.Timestamp("2024-08-01", tz="UTC")  # exclusivo

_TEMP_SEED = 1
_DEMAND_SEED = 7


def _full_grid() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Rejilla UTC regular completa y su expresion en hora de pared de Madrid.

    Returns
    -------
    utc_index, local_naive_index
        `utc_index` es una rejilla horaria regular en UTC, sin huecos por
        construccion. `local_naive_index` es la misma rejilla vista como hora
        local sin huso: en el salto de primavera faltan las horas
        inexistentes; en el vuelco de otono hay marcas repetidas. Es
        exactamente el aspecto que tendria una fuente real que reporta en
        hora local, y es el insumo que usa `SyntheticElectricitySource` para
        que el pipeline de alineado tenga algo genuino que resolver.
    """
    utc_index = pd.date_range(_FULL_START, _FULL_END, freq="h", inclusive="left")
    local_index = utc_index.tz_convert(_NATIVE_TZ).tz_localize(None)
    return utc_index, local_index


def _synthetic_temperature(utc_index: pd.DatetimeIndex) -> np.ndarray:
    """Temperatura horaria determinista con ciclo diurno y anual (Madrid, orientativo).

    No es un modelo climatico: es una curva suave con la forma correcta
    (minimo en enero, maximo en julio; minimo de madrugada, maximo a media
    tarde) mas ruido, calibrada a ojo para que la respuesta termica de las
    series de demanda tenga una relacion en U razonable con la que trabajar.
    """
    rng = np.random.default_rng(_TEMP_SEED)
    local = utc_index.tz_convert(_NATIVE_TZ)
    hour = local.hour.to_numpy().astype(np.float64)
    day_of_year = local.dayofyear.to_numpy().astype(np.float64)

    annual = 16.5 - 10.5 * np.cos(2 * np.pi * (day_of_year - 15.0) / 365.25)
    diurnal = 4.5 * np.cos(2 * np.pi * (hour - 15.0) / 24.0)
    noise = rng.normal(0.0, 1.2, len(utc_index))
    temp: np.ndarray = annual + diurnal + noise
    return temp


def _spain_holiday_flags(local_index: pd.DatetimeIndex) -> np.ndarray:
    """Vector booleano de festivo nacional de Espana, por fecha civil local."""
    years = sorted({ts.year for ts in local_index})
    calendar = holidays_lib.country_holidays("ES", years=years)
    return np.fromiter(
        (ts.date() in calendar for ts in local_index), dtype=bool, count=len(local_index)
    )


# Pesos horarios relativos (no normalizados a 1: son un multiplicador de
# forma, la amplitud absoluta se fija por separado en cada serie).
_RESIDENTIAL_HOURLY_SHAPE = np.array(
    [
        0.55,
        0.45,
        0.40,
        0.38,
        0.40,
        0.50,
        0.65,
        0.85,
        0.95,
        0.85,
        0.75,
        0.72,
        0.80,
        0.85,
        0.80,
        0.78,
        0.85,
        1.00,
        1.25,
        1.45,
        1.55,
        1.35,
        1.05,
        0.75,
    ]
)
_COMMERCIAL_HOURLY_SHAPE = np.array(
    [
        0.10,
        0.08,
        0.07,
        0.07,
        0.08,
        0.12,
        0.25,
        0.55,
        0.90,
        1.15,
        1.30,
        1.35,
        1.20,
        1.25,
        1.35,
        1.30,
        1.20,
        1.10,
        0.90,
        0.55,
        0.30,
        0.20,
        0.15,
        0.12,
    ]
)


def _thermal_response(temp: np.ndarray, *, heating_gain: float, cooling_gain: float) -> np.ndarray:
    """Respuesta termica en forma de U: calefaccion con frio, refrigeracion con calor."""
    base_heat, base_cool = 15.0, 24.0
    heating = np.clip(base_heat - temp, 0.0, None)
    cooling = np.clip(temp - base_cool, 0.0, None)
    response: np.ndarray = heating_gain * heating + cooling_gain * cooling
    return response


def _generate_clean_demand(
    utc_index: pd.DatetimeIndex, local_index: pd.DatetimeIndex, temp: np.ndarray
) -> dict[str, np.ndarray]:
    """Genera las tres series de demanda "verdaderas", antes de inyectar imperfecciones."""
    rng = np.random.default_rng(_DEMAND_SEED)
    n = len(utc_index)
    hour = local_index.hour.to_numpy()
    dow = local_index.dayofweek.to_numpy()
    is_weekend = dow >= 5
    is_holiday = _spain_holiday_flags(local_index)
    is_off = is_weekend | is_holiday
    day_index = ((utc_index - _FULL_START) / pd.Timedelta(days=1)).to_numpy()

    series: dict[str, np.ndarray] = {}

    # -- residential_north: fuerte diaria+semanal, tendencia suave, ruido bajo.
    daily_shape = _RESIDENTIAL_HOURLY_SHAPE[hour]
    flat_shape = np.full(n, _RESIDENTIAL_HOURLY_SHAPE.mean())
    shape = np.where(is_off, 0.7 * daily_shape + 0.3 * flat_shape, daily_shape)
    trend = 120.0 + 0.02 * day_index
    weekly_bonus = np.where(is_off, 1.05, 1.0)
    thermal = _thermal_response(temp, heating_gain=0.9, cooling_gain=0.7)
    noise = rng.normal(0.0, 3.0, n)
    series["residential_north"] = trend + 22.0 * shape * weekly_bonus + thermal + noise

    # -- commercial_mixed: horario comercial, casi cerrado el fin de semana,
    # un par de escalones de nivel (apertura de local nuevo), ruido moderado.
    daily_shape = _COMMERCIAL_HOURLY_SHAPE[hour]
    weekend_factor = np.where(is_off, 0.30, 1.0)
    trend = 200.0 + 0.05 * day_index
    level_shifts = np.zeros(n)
    for shift_day, shift_size in ((120.0, 25.0), (280.0, -15.0)):
        level_shifts += np.where(day_index >= shift_day, shift_size, 0.0)
    thermal = _thermal_response(temp, heating_gain=0.5, cooling_gain=0.6)
    noise = rng.normal(0.0, 8.0, n)
    series["commercial_mixed"] = (
        trend + level_shifts + 55.0 * daily_shape * weekend_factor + thermal + noise
    )

    # -- volatile_industrial: casi plana, casi sin respuesta termica, ruido
    # alto, con lotes de produccion que llegan como un proceso de Poisson y
    # elevan el nivel durante unas horas. Debilmente estacional a proposito.
    base = 300.0 + 8.0 * np.sin(2 * np.pi * hour / 24.0)
    thermal = _thermal_response(temp, heating_gain=0.1, cooling_gain=0.1)
    noise = rng.normal(0.0, 25.0, n)
    batches = np.zeros(n)
    n_batches = max(1, n // 250)  # ~ un lote cada ~10 dias en promedio
    batch_starts = rng.integers(0, n, size=n_batches)
    for start_idx in batch_starts:
        duration = int(rng.integers(4, 30))
        magnitude = float(rng.uniform(60.0, 160.0))
        end_idx = min(n, start_idx + duration)
        batches[start_idx:end_idx] += magnitude
    series["volatile_industrial"] = base + thermal + noise + batches

    return series


def _inject_imperfections(
    frame: pd.DataFrame, *, seed: int, missing_rate: float = 0.004, n_duplicates: int = 20
) -> pd.DataFrame:
    """Anade huecos, duplicados con conflicto, un tramo de ceros y picos atipicos.

    Todo deliberado y reproducible: es lo que le da sustancia al perfil de
    calidad de la Fase 3 (docs/ARCHITECTURE.md), en vez de partir de una serie
    perfecta que no tendria nada que reportar.

    Parameters
    ----------
    frame
        Trama larga limpia, ``unique_id``, ``ds`` (hora local, con las
        marcas ya ausentes/duplicadas propias del DST) y ``y``.
    seed
        Semilla de las imperfecciones inyectadas.
    missing_rate
        Fraccion de filas que se eliminan por serie, simulando caidas de
        sensor no relacionadas con el DST.
    n_duplicates
        Numero de filas por serie que se duplican con un valor ligeramente
        distinto, simulando una retransmision.

    Returns
    -------
    pandas.DataFrame
        Trama con huecos adicionales, duplicados, un tramo de ceros en
        ``commercial_mixed`` y picos atipicos, sin ordenar garantizado.
    """
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []

    for unique_id, group in frame.groupby("unique_id", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)

        n_missing = round(n * missing_rate)
        drop_idx = pd.Index(rng.choice(n, size=n_missing, replace=False))
        kept = group.drop(index=drop_idx).reset_index(drop=True)

        dup_idx = pd.Index(rng.choice(len(kept), size=min(n_duplicates, len(kept)), replace=False))
        duplicated_rows = kept.loc[dup_idx].copy()
        duplicated_rows["y"] = duplicated_rows["y"] * rng.uniform(0.97, 1.03, len(duplicated_rows))

        outlier_idx = pd.Index(rng.choice(len(kept), size=8, replace=False))
        kept.loc[outlier_idx, "y"] = kept.loc[outlier_idx, "y"] * rng.uniform(
            4.0, 8.0, len(outlier_idx)
        )

        if unique_id == "commercial_mixed" and len(kept) > 24 * 10:
            zero_start = len(kept) // 2
            zero_slice = kept.index[zero_start : zero_start + 24 * 3]
            kept.loc[zero_slice, "y"] = 0.0

        parts.append(pd.concat([kept, duplicated_rows], ignore_index=True))

    return pd.concat(parts, ignore_index=True)


class SyntheticElectricitySource:
    """`DataSource` sintetica de demanda electrica: tres series de dificultad escalonada.

    Ver el docstring del modulo para el diseno de cada serie. Genera siempre
    la ventana completa fija internamente (14 meses) y filtra al rango
    pedido al final, de modo que dos llamadas con rangos distintos coinciden
    en la parte que se solapa.

    Parameters
    ----------
    seed
        Semilla de las imperfecciones inyectadas (huecos, duplicados, ceros,
        atipicos). La estructura de las series (estacionalidad, tendencia,
        respuesta termica) no depende de `seed`: es fija, para que distintas
        instancias sigan siendo comparables entre si.
    """

    seed: int

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    @property
    def spec(self) -> SourceSpec:
        """Fuente de rol `TARGET`: tres series de demanda sinteticas."""
        return SourceSpec(
            source_id="synthetic_electricity",
            role=Role.TARGET,
            value_columns=("y",),
            freq="h",
            native_tz=_NATIVE_TZ,
            vintage_aware=False,
            id_semantics="serie sintetica de demanda (ver DEMO_SERIES_IDS)",
        )

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Genera la demanda sintetica y devuelve el tramo `[start, end)`.

        La trama devuelta **no esta alineada**: `ds` esta en hora local sin
        huso, puede tener huecos (huecos genuinos inyectados, mas los propios
        del salto de primavera), duplicados (inyectados, mas los propios del
        vuelco de otono) y no viene garantizada ordenada. Es deliberado: el
        proposito de esta fuente es que la notebook de EDA ejercite
        `chronolab.data.align` de verdad, igual que haria con una fuente real.

        Ver `chronolab.data.protocols.DataSource.fetch` para el contrato
        completo. `as_of` no esta soportado.
        """
        if as_of is not None:
            raise VintageNotSupported(
                f"{self.spec.source_id} no admite as_of (no es vintage-aware)"
            )

        utc_index, local_index = _full_grid()
        temp = _synthetic_temperature(utc_index)
        demand = _generate_clean_demand(utc_index, local_index, temp)

        selected = tuple(str(i) for i in ids) if ids is not None else DEMO_SERIES_IDS
        parts = [
            pd.DataFrame({"unique_id": series_id, "ds": local_index, "y": demand[series_id]})
            for series_id in selected
            if series_id in demand
        ]
        frame = pd.concat(parts, ignore_index=True)
        frame = _inject_imperfections(frame, seed=self.seed)

        return frame[(frame["ds"] >= start) & (frame["ds"] < end)].reset_index(drop=True)


class SyntheticWeatherSource:
    """`DataSource` sintetica de temperatura horaria, consistente con la demanda.

    Usa la misma `_synthetic_temperature` que alimenta la respuesta termica de
    `SyntheticElectricitySource`, de modo que la relacion demanda-temperatura
    de la Fase 3 sea genuina y no una coincidencia fabricada por separado.
    A diferencia de la demanda, se devuelve directamente en UTC (como
    `OpenMeteoSource` con ``timezone=UTC``): no hay DST que resolver en esta
    fuente.
    """

    @property
    def spec(self) -> SourceSpec:
        """Fuente de rol `FUTR_EXOG`: temperatura sintetica de una unica ubicacion."""
        return SourceSpec(
            source_id="synthetic_weather",
            role=Role.FUTR_EXOG,
            value_columns=("temp_c",),
            freq="h",
            native_tz="UTC",
            vintage_aware=False,
            id_semantics="ubicacion sintetica unica",
        )

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Genera temperatura horaria sintetica y devuelve el tramo `[start, end)`.

        Ver `chronolab.data.protocols.DataSource.fetch`. `as_of` no esta
        soportado.
        """
        if as_of is not None:
            raise VintageNotSupported(
                f"{self.spec.source_id} no admite as_of (no es vintage-aware)"
            )

        utc_index, _ = _full_grid()
        temp = _synthetic_temperature(utc_index)
        frame = pd.DataFrame(
            {"unique_id": "madrid_demo", "ds": utc_index.tz_localize(None), "temp_c": temp}
        )
        return frame[(frame["ds"] >= start) & (frame["ds"] < end)].reset_index(drop=True)
