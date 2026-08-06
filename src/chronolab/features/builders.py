"""Conjuntos de features con nombre usados por los modelos de machine learning.

Tres familias, y una regla de reparto de responsabilidades entre este modulo y
`chronolab.models.adapters.mlforecast` que conviene tener presente antes de leer
el resto:

1. **Lags, ventanas moviles y diferencias de la propia objetivo** (``y``) no se
   calculan aqui. `mlforecast` sabe generarlos y, sobre todo, sabe
   **recomputarlos en la prediccion recursiva realimentando sus propias
   predicciones** cuando un lag corto (``lag(y, 1)``) hace falta mas alla de su
   `max_lead`; reimplementar eso a mano es la fuente de bugs numero uno de
   cualquier proyecto de forecasting con ML. Este modulo se limita a declarar
   **los numeros** —que lags, que ventanas, que diferencias— en
   `TargetFeatureConfig`, un DTO sin ninguna referencia a la libreria. El
   adaptador (`models.adapters.mlforecast`) es quien traduce ese DTO a
   `lags=`/`lag_transforms=` de `mlforecast` y quien decide, segun la
   estrategia (recursiva o directa via `max_horizon`), como usarlos.
2. **Calendario y termicas son features exogenas de verdad**, calculadas aqui
   con las primitivas causales de `chronolab.features.ops` y de
   `chronolab.data.calendar`, cada una con su `max_lead` calculado —nunca
   declarado— por `chronolab.features.roles`. `mlforecast` no sabe generar
   estas por su cuenta (no son funcion de la propia objetivo), asi que viajan
   como columnas regresoras dinamicas: se anaden al `DataFrame` de
   entrenamiento y hay que suministrarlas tambien en el tramo de prediccion.
3. **El calendario es siempre utilizable en cualquier horizonte**
   (``max_lead = UNBOUNDED``: es funcion determinista de ``ds``); las termicas
   heredan el `max_lead` que tenga la columna de temperatura declarada en el
   `PanelSpec` del panel (`futr_exog` si hay prevision, `hist_exog` si solo se
   observa). `select_usable` aplica el filtrado de
   `chronolab.features.roles.select_for_lead` sin que el llamante tenga que
   conocer el algebra.

Todas las features de este modulo son estrictamente retrospectivas: ninguna
mira mas alla del instante en que se calcula. Las termicas se apoyan en
`features.ops.lag`, que ya lleva esa garantia; las de calendario son funcion
pura de `ds`, sin ningun valor observado de por medio.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from chronolab.data.calendar import calendar_features, fourier_terms
from chronolab.features.ops import Feature, RollStat, from_column, lag
from chronolab.features.roles import UNBOUNDED, FeatureSpec, MaxLead, select_for_lead
from chronolab.panel import Panel

__all__ = [
    "CALENDAR_FOURIER_PERIODS",
    "DEFAULT_THERMAL_FEATURES",
    "LAGS",
    "ROLL_STATS",
    "ROLL_WINDOWS",
    "TargetFeatureConfig",
    "ThermalFeatureConfig",
    "calendar_feature_set",
    "feature_frame",
    "feature_set",
    "select_usable",
    "thermal_feature_set",
]

LAGS: tuple[int, ...] = (1, 2, 3, 24, 48, 168, 336)
"""Retardos canonicos de la objetivo, en pasos horarios: hasta dos semanas."""

ROLL_WINDOWS: tuple[int, ...] = (24, 168)
"""Ventanas moviles canonicas, en pasos: ciclo diario y semanal."""

ROLL_STATS: tuple[RollStat, ...] = ("mean", "std", "min", "max")
"""Estadisticos canonicos sobre cada ventana movil."""

CALENDAR_FOURIER_PERIODS: Mapping[str, float] = {
    "daily": 24.0,
    "weekly": 168.0,
    "annual": 365.25 * 24.0,
}
"""Periodos, en horas, de los terminos de Fourier de calendario (§ enunciado)."""


# --------------------------------------------------------------------------- #
# Numeros de la propia objetivo: DTO, sin referencia a mlforecast
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TargetFeatureConfig:
    """Numeros canonicos de lags, ventanas moviles y diferencias sobre ``y``.

    Declara **que** derivar de la objetivo, nunca **como** calcularlo: no hay
    aqui ninguna llamada a `features.ops` ni a `mlforecast`. Es
    `models.adapters.mlforecast` quien lee esta configuracion y la traduce a
    `lags=` y `lag_transforms=`, dejando que la libreria gestione tanto el
    calculo como la recursividad (ver el docstring del modulo).

    Parameters
    ----------
    lags
        Retardos de la objetivo, en pasos, mayores o iguales que uno.
    roll_windows
        Longitudes de ventana movil, en pasos, mayores o iguales que uno.
    roll_stats
        Estadisticos a calcular sobre cada ventana.
    roll_shift
        Desplazamiento de las ventanas moviles: terminan en
        ``t - roll_shift``, nunca en ``t``.
    diff_lags
        Ordenes, en pasos, de las diferencias simples ``y_{t-s} - y_{t-s-k}``
        a incluir, con ``s = diff_shift``.
    pct_change_lags
        Retardos, en pasos, sobre los que calcular una tasa de cambio
        ``(y_{t-s} - y_{t-s-k}) / y_{t-s-k}``, con ``s = diff_shift``.
    diff_shift
        Desplazamiento comun de diferencias y tasas de cambio: nunca usan el
        valor en ``t``, igual que las ventanas moviles.

    Raises
    ------
    ValueError
        Si algun lag, ventana u orden de diferencia es menor que uno.
    """

    lags: tuple[int, ...] = LAGS
    roll_windows: tuple[int, ...] = ROLL_WINDOWS
    roll_stats: tuple[RollStat, ...] = ROLL_STATS
    roll_shift: int = 1
    diff_lags: tuple[int, ...] = (1, 24)
    pct_change_lags: tuple[int, ...] = (24, 168)
    diff_shift: int = 1

    def __post_init__(self) -> None:
        """Valida que todos los numeros declarados son pasos positivos."""
        if any(k < 1 for k in self.lags):
            raise ValueError(f"cada lag debe ser >= 1: {self.lags}")
        if any(w < 1 for w in self.roll_windows):
            raise ValueError(f"cada ventana debe ser >= 1: {self.roll_windows}")
        if self.roll_shift < 1:
            raise ValueError(f"roll_shift debe ser >= 1: {self.roll_shift}")
        if any(k < 1 for k in (*self.diff_lags, *self.pct_change_lags)):
            raise ValueError(
                f"cada orden de diferencia debe ser >= 1: diff_lags={self.diff_lags}, "
                f"pct_change_lags={self.pct_change_lags}"
            )
        if self.diff_shift < 1:
            raise ValueError(f"diff_shift debe ser >= 1: {self.diff_shift}")


DEFAULT_TARGET_FEATURES = TargetFeatureConfig()
"""Instancia con los numeros canonicos del proyecto."""


# --------------------------------------------------------------------------- #
# Calendario: siempre disponible, funcion determinista de `ds`
# --------------------------------------------------------------------------- #


def _from_series(values: pd.Series, *, max_lead: MaxLead = UNBOUNDED) -> Feature:
    """Envuelve una serie ya calculada como `Feature`, sin pasar por `features.ops`.

    Legitimo unicamente para columnas cuyo valor en ``t`` no depende de ningun
    dato observado —el calendario— o que son una transformacion puntual de
    otra `Feature` que no desplaza nada en el tiempo —los grados-dia sobre la
    temperatura, mas abajo—: en ambos casos el `max_lead` de origen no cambia,
    asi que se pasa tal cual en lugar de recalcularse con el algebra de
    `chronolab.features.roles`, que esta pensada para retardos y ventanas, no
    para funciones puntuales.

    Parameters
    ----------
    values
        Serie ya nombrada.
    max_lead
        Adelanto a declarar. `UNBOUNDED` por defecto, el caso de calendario.

    Returns
    -------
    Feature
    """
    return Feature(values=values, spec=FeatureSpec(name=str(values.name), max_lead=max_lead))


def calendar_feature_set(
    panel: Panel,
    *,
    country: str | None = None,
    subdiv: str | None = None,
    fourier_periods: Mapping[str, float] = CALENDAR_FOURIER_PERIODS,
    fourier_order: int = 2,
) -> tuple[Feature, ...]:
    """Calendario: hora, dia de semana, dia del mes, mes, festivos y Fourier.

    Todas con ``max_lead = UNBOUNDED``: cada una es funcion determinista de
    ``ds`` (docs/ARCHITECTURE.md §4.4, fila "Features de calendario derivadas
    de ds"), nunca del valor observado de ninguna serie, asi que se conocen
    para cualquier instante futuro que se quiera predecir, sea cual sea el
    horizonte o la estrategia.

    Parameters
    ----------
    panel
        Panel de referencia. Solo se usan su columna ``ds`` y
        ``spec.tz_display`` —"es festivo" y "es la hora punta" son
        propiedades del reloj de pared local, no de UTC (docs/ARCHITECTURE.md
        §3.3).
    country, subdiv
        Ver `chronolab.data.calendar.holiday_flags`. Sin `country` no se
        generan ``is_holiday`` ni ``is_holiday_eve`` (vispera de festivo).
    fourier_periods
        Periodos, en horas, de los terminos de Fourier. Por defecto diario,
        semanal y anual (`CALENDAR_FOURIER_PERIODS`), las tres
        estacionalidades que pide el enunciado del proyecto.
    fourier_order
        Armonicos por periodo.

    Returns
    -------
    tuple of Feature
        Una por columna de calendario (``hour``, ``dayofweek``, ``day``,
        ``month``, ``is_weekend``, los pares seno/coseno, y si `country` no es
        ``None``, ``is_holiday`` e ``is_holiday_eve``) mas una por columna de
        Fourier generada.
    """
    ds = panel.df["ds"]
    calendar = calendar_features(
        ds, tz_display=panel.spec.tz_display, country=country, subdiv=subdiv
    )
    fourier = fourier_terms(ds, periods=fourier_periods, order=fourier_order)

    features = [_from_series(calendar[column]) for column in calendar.columns if column != "ds"]
    features += [_from_series(fourier[column]) for column in fourier.columns if column != "ds"]
    return tuple(features)


# --------------------------------------------------------------------------- #
# Termicas: temperatura, grados-dia y sus lags
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ThermalFeatureConfig:
    """Numeros de las features termicas: temperatura, grados-dia y sus retardos.

    Parameters
    ----------
    temp_column
        Nombre de la columna de temperatura en ``panel.spec.value_columns``.
    heating_base, cooling_base
        Temperaturas base de los grados-dia de calefaccion (``HDD``) y
        refrigeracion (``CDD``), en grados Celsius. 18 C por defecto: el
        estandar habitual en climatologia energetica.
    lags
        Retardos, en pasos, que se anaden de la temperatura y de cada
        grado-dia.

    Raises
    ------
    ValueError
        Si algun lag es menor que uno.
    """

    temp_column: str = "temp_c"
    heating_base: float = 18.0
    cooling_base: float = 18.0
    lags: tuple[int, ...] = (1, 24, 168)

    def __post_init__(self) -> None:
        """Valida que los retardos declarados son pasos positivos."""
        if any(k < 1 for k in self.lags):
            raise ValueError(f"cada lag debe ser >= 1: {self.lags}")


DEFAULT_THERMAL_FEATURES = ThermalFeatureConfig()
"""Instancia con los numeros canonicos: temperatura ``temp_c``, base 18 C."""


def _degree_days(temp: Feature, *, base: float, kind: Literal["heating", "cooling"]) -> Feature:
    """Grados-dia de calefaccion o refrigeracion, punto a punto sobre la temperatura.

    ``max(base - temp, 0)`` para calefaccion, ``max(temp - base, 0)`` para
    refrigeracion, evaluados en el **mismo instante** que `temp`: no
    desplazan nada en el tiempo, asi que heredan su `max_lead` sin cambios
    (ver `_from_series`).

    Parameters
    ----------
    temp
        Feature de temperatura, tipicamente de `chronolab.features.ops.from_column`.
    base
        Temperatura base, en grados Celsius.
    kind
        ``"heating"`` (HDD) o ``"cooling"`` (CDD).

    Returns
    -------
    Feature
        Nombrada ``"<temp>_hdd<base>"`` o ``"<temp>_cdd<base>"``.
    """
    raw = temp.values.to_numpy(dtype=float)
    if kind == "heating":
        degrees, suffix = np.clip(base - raw, 0.0, None), "hdd"
    else:
        degrees, suffix = np.clip(raw - base, 0.0, None), "cdd"

    name = f"{temp.name}_{suffix}{base:g}"
    values = pd.Series(degrees.astype(np.float32), index=temp.values.index, name=name)
    return _from_series(values, max_lead=temp.max_lead)


def thermal_feature_set(
    panel: Panel, config: ThermalFeatureConfig = DEFAULT_THERMAL_FEATURES
) -> tuple[Feature, ...]:
    """Temperatura, grados-dia de calefaccion/refrigeracion y sus retardos.

    El `max_lead` de todo el conjunto se deduce del rol declarado para
    `config.temp_column` en ``panel.spec`` (docs/ARCHITECTURE.md §4.4): si la
    temperatura es `futr_exog` (hay prevision archivada o simulada), el
    conjunto entero queda ilimitado; si es `hist_exog` (solo observada), cada
    retardo queda utilizable hasta ese numero de pasos y `select_usable` —o el
    motor de backtesting, via `chronolab.features.roles.select_for_lead`— lo
    filtra por adelanto exactamente igual que cualquier otra feature derivada.

    Parameters
    ----------
    panel
        Panel de referencia. `config.temp_column` debe tener un rol declarado
        en ``panel.spec``.
    config
        Numeros de la construccion; ver `ThermalFeatureConfig`.

    Returns
    -------
    tuple of Feature
        Temperatura, HDD y CDD sin desplazar, mas ``len(config.lags)``
        retardos de cada una de las tres: ``3 * (1 + len(config.lags))``
        features en total.

    Raises
    ------
    KeyError
        Si `config.temp_column` no tiene rol declarado en ``panel.spec``.
    """
    temp = from_column(panel, config.temp_column)
    hdd = _degree_days(temp, base=config.heating_base, kind="heating")
    cdd = _degree_days(temp, base=config.cooling_base, kind="cooling")

    features = [temp, hdd, cdd]
    for base_feature in (temp, hdd, cdd):
        features.extend(lag(panel, base_feature, k) for k in config.lags)
    return tuple(features)


# --------------------------------------------------------------------------- #
# Composicion y utilidades para los adaptadores
# --------------------------------------------------------------------------- #


def feature_set(
    panel: Panel,
    *,
    country: str | None = None,
    subdiv: str | None = None,
    fourier_periods: Mapping[str, float] = CALENDAR_FOURIER_PERIODS,
    fourier_order: int = 2,
    thermal: ThermalFeatureConfig | None = DEFAULT_THERMAL_FEATURES,
) -> tuple[Feature, ...]:
    """Calendario mas, opcionalmente, termicas: las exogenas manuales completas.

    Es el conjunto que `chronolab.models.adapters.mlforecast` anade como
    regresoras dinamicas, ademas de los lags, ventanas y diferencias de la
    propia objetivo que gestiona mlforecast a partir de `TargetFeatureConfig`.

    Parameters
    ----------
    panel
        Panel de referencia.
    country, subdiv, fourier_periods, fourier_order
        Ver `calendar_feature_set`.
    thermal
        Configuracion de `thermal_feature_set`, o ``None`` para omitir las
        termicas —paneles sin exogena de temperatura declarada.

    Returns
    -------
    tuple of Feature
        Calendario seguido de termicas, si las hay.
    """
    features = list(
        calendar_feature_set(
            panel,
            country=country,
            subdiv=subdiv,
            fourier_periods=fourier_periods,
            fourier_order=fourier_order,
        )
    )
    if thermal is not None:
        features.extend(thermal_feature_set(panel, thermal))
    return tuple(features)


def select_usable(
    features: tuple[Feature, ...], lead: int, *, supports_recursive: bool = False
) -> tuple[Feature, ...]:
    """Filtra `features` a las utilizables para un adelanto dado desde el cutoff.

    Envoltorio fino sobre `chronolab.features.roles.select_for_lead`, que
    opera sobre `FeatureSpec` y descarta los valores: este conserva ambos, que
    es lo que necesita quien va a montar la matriz de diseno.

    Parameters
    ----------
    features
        Candidatas, tipicamente la salida de `feature_set`.
    lead
        Adelanto real desde el cutoff (``gap + h_step``). Cuando una unica
        configuracion debe servir a todo un horizonte de una vez —como hace
        `mlforecast` con ``max_horizon`` en la estrategia directa— se pasa el
        adelanto **maximo** del plan, el caso mas restrictivo: una feature que
        no sobrevive al ultimo paso no puede formar parte de un conjunto de
        regresoras compartido por todos los pasos.
    supports_recursive
        Ver `chronolab.features.roles.select_for_lead`. Ninguna de las
        features de este modulo se marca ``recursive_only``: la recursion
        solo tiene sentido sobre la propia objetivo, y esa la gestiona
        integramente `mlforecast`, nunca una exogena de este modulo.

    Returns
    -------
    tuple of Feature
        Subconjunto admisible, en el orden de entrada.
    """
    specs = tuple(feature.spec for feature in features)
    allowed = {
        spec.name for spec in select_for_lead(specs, lead, supports_recursive=supports_recursive)
    }
    return tuple(feature for feature in features if feature.name in allowed)


def feature_frame(panel: Panel, features: tuple[Feature, ...]) -> pd.DataFrame:
    """Ensambla `features` en una trama ``unique_id, ds, <feature...>`` alineada al panel.

    Es el formato que espera `mlforecast` para sus regresoras dinamicas: una
    columna por feature, alineada por posicion con ``panel.df``.

    Parameters
    ----------
    panel
        Panel del que proceden las features.
    features
        Features ya calculadas, con el mismo indice que ``panel.df``.

    Returns
    -------
    pandas.DataFrame
        ``unique_id``, ``ds`` y una columna por feature, en el orden de
        entrada.

    Raises
    ------
    ValueError
        Si dos features comparten nombre: montar la matriz de diseno con una
        columna duplicada silenciaria cual de las dos sobrevive al `merge`.
    """
    names = [feature.name for feature in features]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"nombres de feature repetidos: {duplicates}")

    frame = panel.df[["unique_id", "ds"]].copy()
    for feature in features:
        frame[feature.name] = feature.values.to_numpy()
    return frame
