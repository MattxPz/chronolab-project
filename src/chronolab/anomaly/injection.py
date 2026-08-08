"""Inyeccion de anomalias sinteticas tipadas y generacion del ground truth.

Seis tipos con magnitud y duracion parametrizadas: pico puntual, cambio de
nivel (escalon), cambio de varianza, desfase estacional, congelacion del
sensor y hueco de datos. `docs/ARCHITECTURE.md` §7.4 documenta cinco -el
hueco de datos es una amplicacion de este modulo, y se anota aqui en vez de
en silencio; encaja sin fricción porque el invariante I3 del panel ya trata
un hueco como ``y = NaN`` en una fila que sigue existiendo, exactamente lo que
esta anomalia produce.

**Donde vive la barrera de fuga.** No hay ninguna: este modulo se ejecuta
*antes* de que exista ningun `Panel` recortado por ventana, sobre datos que
ya son completamente conocidos por construccion (son sinteticos). No inyecta
nada despues del hecho -no reescribe `y_hat` ni cuantiles-, asi que un modelo
que se entrene sobre el panel contaminado ve exactamente lo que veria un
sistema real con un sensor que falla: la anomalia esta en la unica fuente de
verdad, `y`, y de ahi se propaga a residuos, forecasts y scores por el camino
normal.

**Cada especificacion se mide contra un contexto local, no global.** La
magnitud de las cinco anomalias que desplazan un valor (todas salvo el hueco
de datos) se expresa en desviaciones tipicas *locales*: la mediana y la MAD
(escalada para estimar sigma bajo normalidad) de los `reference_window`
puntos que preceden a `start`, en la serie tal y como esta en el momento de
aplicar esa especificacion -no en el panel original si una especificacion
anterior ya la toco-. Es deliberado: una serie con tendencia o con otra
anomalia ya inyectada cerca no debe hacer que la siguiente inyeccion se mida
contra una escala que ya no describe el tramo que la precede.

**La fuente de `seasonal_phase` es siempre el panel original**, nunca el ya
mutado: desplazar la fase leyendo de una serie que otra especificacion ya
alterase compondria dos anomalias en una sin que ninguna de las dos quedase
etiquetada como lo que realmente es.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from chronolab.panel import Panel

__all__ = ["KINDS", "TRUTH_COLUMNS", "AnomalyKind", "AnomalySpec", "inject_anomalies"]

AnomalyKind = Literal[
    "spike",
    "level_shift",
    "variance_shift",
    "seasonal_phase",
    "sensor_freeze",
    "data_gap",
]

KINDS: tuple[AnomalyKind, ...] = (
    "spike",
    "level_shift",
    "variance_shift",
    "seasonal_phase",
    "sensor_freeze",
    "data_gap",
)
"""Los seis tipos que admite este modulo, en el orden en que se documentan."""

TRUTH_COLUMNS: tuple[str, ...] = (
    "unique_id",
    "ds",
    "is_anomaly",
    "event_id",
    "anomaly_type",
    "severity",
    "injection_seed",
)
"""Columnas de `datasets/dataset_id=<d>/anomaly_truth.parquet` (docs/ARCHITECTURE.md §7.4).

La tabla es **dispersa**: solo lleva las filas efectivamente alteradas, todas
con ``is_anomaly=True``. Una fila ``unique_id, ds`` ausente de esta tabla es
normal por convenio; reconstruir la mascara densa es cosa de un `outer join`
contra la rejilla del panel, no de esta tabla.
"""

_EPS = 1e-8


@dataclass(frozen=True, slots=True)
class AnomalySpec:
    """Una anomalia sintetica a inyectar: tipo, serie, tramo y magnitud.

    Parameters
    ----------
    kind
        Uno de `KINDS`.
    unique_id
        Serie afectada. Debe existir en el panel al inyectar.
    start
        Primer instante del tramo, en la rejilla del panel.
    duration
        Longitud del tramo en pasos de rejilla, ``>= 1``.
    magnitude
        Semantica segun `kind`:

        - ``spike``, ``level_shift``, ``sensor_freeze``: desviaciones tipicas
          locales del desplazamiento (o del ruido residual sobre el valor
          congelado, para `sensor_freeze`).
        - ``variance_shift``: factor multiplicativo (``> 0``) sobre la
          desviacion respecto de la media local. ``> 1`` infla la varianza,
          ``< 1`` la comprime.
        - ``seasonal_phase``: desfase en pasos de rejilla (``>= 1``); el tramo
          se sustituye por los valores originales de ``start - magnitude``
          pasos en adelante.
        - ``data_gap``: fraccion de puntos del tramo que se convierten en
          `NaN`, en ``(0, 1]``.
    direction
        Solo para ``spike`` y ``level_shift``: ``"up"``, ``"down"`` o
        ``"random"`` (una moneda por punto en `spike`, una sola tirada para
        todo el tramo en `level_shift`).

    Raises
    ------
    ValueError
        Si `kind` no es uno de `KINDS`, si `duration` es menor que uno, si
        `magnitude` no es finita, o si `magnitude` incumple el dominio
        especifico del tipo.
    """

    kind: AnomalyKind
    unique_id: str
    start: pd.Timestamp
    duration: int
    magnitude: float
    direction: Literal["up", "down", "random"] = "random"

    def __post_init__(self) -> None:
        """Valida la especificacion."""
        if self.kind not in KINDS:
            raise ValueError(f"kind debe ser uno de {KINDS}: {self.kind!r}")
        if self.duration < 1:
            raise ValueError(f"duration debe ser >= 1: {self.duration}")
        if not math.isfinite(self.magnitude):
            raise ValueError(f"magnitude debe ser finita: {self.magnitude}")
        if self.kind == "data_gap" and not 0.0 < self.magnitude <= 1.0:
            raise ValueError(
                f"data_gap: magnitude es la fraccion de puntos borrados, en (0, 1]: "
                f"{self.magnitude}"
            )
        if self.kind == "variance_shift" and self.magnitude <= 0.0:
            raise ValueError(f"variance_shift: magnitude debe ser > 0: {self.magnitude}")
        if self.kind == "seasonal_phase" and self.magnitude < 1.0:
            raise ValueError(f"seasonal_phase: magnitude (pasos) debe ser >= 1: {self.magnitude}")


def inject_anomalies(
    panel: Panel,
    specs: Sequence[AnomalySpec],
    *,
    seed: int = 0,
    reference_window: int | None = None,
) -> tuple[Panel, pd.DataFrame]:
    """Aplica las anomalias declaradas sobre una copia del panel y genera su verdad.

    Las especificaciones se aplican **en el orden dado**. Si dos se solapan en
    ``(unique_id, ds)``, la ultima gana y la primera desaparece de la tabla de
    verdad para esos puntos: es la misma semantica de "ultima escritura gana"
    que cualquier asignacion secuencial, y se documenta en vez de prohibirse
    porque encadenar anomalias -un hueco de datos sobre un cambio de nivel- es
    un caso de prueba legitimo.

    Parameters
    ----------
    panel
        Panel de partida. No se modifica.
    specs
        Anomalias a inyectar, en el orden en que se aplican.
    seed
        Semilla del generador que decide los signos aleatorios de `spike` y
        `level_shift` y los puntos concretos de `data_gap`. Un mismo `specs`
        con la misma `seed` produce exactamente el mismo panel y la misma
        verdad.
    reference_window
        Puntos de contexto retrospectivo para la mediana y la MAD locales.
        Por defecto, dos veces la estacionalidad mas corta del panel
        (``2 * spec.mase_season``), el mismo criterio que usa
        `chronolab.anomaly.conformal.ConformalDetector.max_freeze`.

    Returns
    -------
    tuple
        Panel contaminado (mismo `spec`, misma `static`) y la tabla de verdad
        con las columnas `TRUTH_COLUMNS`, ordenada por ``(unique_id, ds)``.

    Raises
    ------
    ValueError
        Si alguna especificacion referencia una serie ausente del panel, si
        su tramo cae fuera del rango observado de esa serie, o si `start` no
        cae en la rejilla del panel.
    """
    rng = np.random.default_rng(seed)
    window = reference_window if reference_window is not None else 2 * panel.spec.mase_season
    if window < 1:
        raise ValueError(f"reference_window debe ser >= 1: {window}")

    df = panel.df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    original = df.set_index(["unique_id", "ds"])["y"]

    truth_rows: list[dict[str, object]] = []
    for spec in specs:
        span = _span(spec, panel=panel, df=df)
        mask = (df["unique_id"] == spec.unique_id) & df["ds"].isin(span)
        current = df.loc[mask, "y"].to_numpy(dtype=float)

        local_mean, local_std = _local_reference(
            df, unique_id=spec.unique_id, before=spec.start, window=window
        )
        new_values, severity, affected = _apply(
            spec,
            current=current,
            local_mean=local_mean,
            local_std=local_std,
            rng=rng,
            original=original,
            span=span,
            freq=panel.spec.freq,
        )

        # `y` viaja en `float32` (invariante I5); una asignacion en `float64`
        # sin convertir falla en pandas recientes en vez de degradar en
        # silencio, asi que la conversion es explicita.
        df.loc[mask, "y"] = new_values.astype(np.float32)
        truth_rows.extend(
            _truth_rows(spec, span=span, severity=severity, affected=affected, seed=seed)
        )

    truth = (
        pd.DataFrame(truth_rows)[list(TRUTH_COLUMNS)]
        if truth_rows
        else pd.DataFrame({name: pd.Series(dtype="object") for name in TRUTH_COLUMNS})
    )
    truth = truth.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    contaminated = Panel(df=df, spec=panel.spec, static=panel.static)
    return contaminated, truth


def _span(spec: AnomalySpec, *, panel: Panel, df: pd.DataFrame) -> pd.DatetimeIndex:
    """Rejilla del tramo de una especificacion, validada contra el panel.

    Parameters
    ----------
    spec
        Especificacion a validar.
    panel
        Panel de partida.
    df
        `panel.df` ya ordenado.

    Returns
    -------
    pandas.DatetimeIndex
        ``spec.duration`` marcas de tiempo consecutivas a `spec.start`.

    Raises
    ------
    ValueError
        Si la serie no existe, si `start` no cae en la rejilla, o si el tramo
        se sale del rango observado de esa serie.
    """
    own = df.loc[df["unique_id"] == spec.unique_id, "ds"]
    if own.empty:
        raise ValueError(f"'{spec.unique_id}' no esta en el panel")

    span = pd.date_range(spec.start, periods=spec.duration, freq=panel.spec.freq)
    first, last = own.min(), own.max()
    if span[0] < first or span[-1] > last:
        raise ValueError(
            f"el tramo de '{spec.kind}' para '{spec.unique_id}' ({span[0]}..{span[-1]}) "
            f"se sale del rango observado de la serie ({first}..{last})"
        )
    if not span.isin(own.to_numpy()).all():
        raise ValueError(f"start={spec.start} no cae en la rejilla de '{spec.unique_id}'")
    return span


def _local_reference(
    df: pd.DataFrame, *, unique_id: str, before: pd.Timestamp, window: int
) -> tuple[float, float]:
    """Mediana y MAD (escalada) de los `window` puntos anteriores a `before`.

    Se leen del `df` **en su estado actual**, es decir, ya con las
    especificaciones anteriores aplicadas: es lo que veria un detector que
    solo mira el pasado, y es la razon de que dos anomalias cercanas no se
    midan contra la misma escala si la primera ya desplazo el contexto de la
    segunda.

    Parameters
    ----------
    df
        Panel en su estado actual, ordenado.
    unique_id
        Serie.
    before
        Instante estrictamente anterior al que se toma el contexto.
    window
        Puntos de contexto.

    Returns
    -------
    tuple
        ``(mediana, mad_escalada)``. ``(0.0, 1.0)`` si no hay contexto finito.
    """
    own = df.loc[(df["unique_id"] == unique_id) & (df["ds"] < before)].sort_values("ds")
    values = own["y"].tail(window).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) * 1.4826
    return median, max(mad, _EPS)


def _apply(
    spec: AnomalySpec,
    *,
    current: np.ndarray,
    local_mean: float,
    local_std: float,
    rng: np.random.Generator,
    original: pd.Series,
    span: pd.DatetimeIndex,
    freq: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica la transformacion del tipo declarado.

    Parameters
    ----------
    spec
        Especificacion.
    current
        Valores actuales del tramo (tras especificaciones previas).
    local_mean, local_std
        Referencia local de `_local_reference`.
    rng
        Generador compartido de toda la inyeccion.
    original
        `y` del panel **original**, indexado por ``(unique_id, ds)``. Solo lo
        usa `seasonal_phase`.
    span
        Rejilla del tramo.
    freq
        Alias de offset de la rejilla del panel. Solo lo usa `seasonal_phase`.

    Returns
    -------
    tuple
        ``(nuevos_valores, severidad, afectado)``, los tres de la misma
        longitud que `current`. ``severidad`` es ``NaN`` donde no aplica
        (`data_gap`) o donde el punto no se vio afectado.
    """
    if spec.kind == "spike":
        signs = _signs(spec.direction, size=current.size, rng=rng)
        new_values = current + signs * spec.magnitude * local_std
        affected = np.ones(current.size, dtype=bool)
    elif spec.kind == "level_shift":
        sign = _sign(spec.direction, rng)
        new_values = current + sign * spec.magnitude * local_std
        affected = np.ones(current.size, dtype=bool)
    elif spec.kind == "variance_shift":
        new_values = local_mean + (current - local_mean) * spec.magnitude
        affected = np.ones(current.size, dtype=bool)
    elif spec.kind == "seasonal_phase":
        new_values, affected = _seasonal_phase(
            spec, original=original, span=span, fallback=current, freq=freq
        )
    elif spec.kind == "sensor_freeze":
        frozen = current[0] if current.size else local_mean
        noise = rng.normal(0.0, spec.magnitude * local_std, size=current.size)
        new_values = np.full(current.size, frozen) + noise
        affected = np.ones(current.size, dtype=bool)
    else:  # data_gap
        new_values = current.copy()
        n_drop = max(1, round(spec.magnitude * current.size))
        dropped = rng.choice(current.size, size=min(n_drop, current.size), replace=False)
        affected = np.zeros(current.size, dtype=bool)
        affected[dropped] = True
        new_values[dropped] = np.nan

    severity = np.full(current.size, np.nan, dtype=float)
    if spec.kind != "data_gap":
        with np.errstate(invalid="ignore"):
            severity = np.abs(new_values - current) / local_std
    return new_values, severity, affected


def _seasonal_phase(
    spec: AnomalySpec,
    *,
    original: pd.Series,
    span: pd.DatetimeIndex,
    fallback: np.ndarray,
    freq: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sustituye el tramo por los valores originales desplazados en fase.

    Parameters
    ----------
    spec
        Especificacion, con `magnitude` en pasos de rejilla.
    original
        `y` del panel original, indexado por ``(unique_id, ds)``.
    span
        Rejilla del tramo.
    fallback
        Valores actuales, usados donde la fuente desplazada no existe.
    freq
        Alias de offset de la rejilla del panel (``panel.spec.freq``). Se pide
        explicito -en vez de leer ``span.freq``, que pandas tipa como
        opcional- para que el desplazamiento sea un offset siempre valido.

    Returns
    -------
    tuple
        ``(nuevos_valores, afectado)``. Un punto queda sin afectar -y
        conserva su valor- si la fuente desplazada cae fuera del panel o es
        `NaN`.
    """
    shift = round(spec.magnitude)
    offset = pd.tseries.frequencies.to_offset(freq)
    source = span - shift * offset
    new_values = fallback.copy()
    affected = np.zeros(len(span), dtype=bool)
    for i, ts in enumerate(source):
        key = (spec.unique_id, ts)
        if key not in original.index:
            continue
        value = original.loc[key]
        if not np.isfinite(value):
            continue
        new_values[i] = float(value)
        affected[i] = True
    return new_values, affected


def _sign(direction: Literal["up", "down", "random"], rng: np.random.Generator) -> float:
    """Un signo, fijo o sorteado, para todo un tramo.

    Parameters
    ----------
    direction
        ``"up"``, ``"down"`` o ``"random"``.
    rng
        Generador compartido.

    Returns
    -------
    float
        ``1.0`` o ``-1.0``.
    """
    if direction == "up":
        return 1.0
    if direction == "down":
        return -1.0
    return float(rng.choice([-1.0, 1.0]))


def _signs(
    direction: Literal["up", "down", "random"], *, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Un signo por punto, fijo o sorteado independientemente.

    Parameters
    ----------
    direction
        ``"up"``, ``"down"`` o ``"random"``.
    size
        Numero de puntos.
    rng
        Generador compartido.

    Returns
    -------
    numpy.ndarray
        ``float64`` de `size` valores en ``{-1.0, 1.0}``.
    """
    if direction == "up":
        return np.ones(size, dtype=float)
    if direction == "down":
        return -np.ones(size, dtype=float)
    result: np.ndarray = rng.choice([-1.0, 1.0], size=size)
    return result


def _truth_rows(
    spec: AnomalySpec,
    *,
    span: pd.DatetimeIndex,
    severity: np.ndarray,
    affected: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    """Filas de la tabla de verdad para los puntos que una especificacion afecto de verdad.

    Parameters
    ----------
    spec
        Especificacion aplicada.
    span
        Rejilla del tramo.
    severity
        Severidad por punto, de `_apply`.
    affected
        Mascara de puntos realmente alterados.
    seed
        Semilla de la inyeccion completa.

    Returns
    -------
    list of dict
        Una fila por punto con ``affected=True``.
    """
    event_id = f"{spec.kind}|{spec.unique_id}|{span[0]:%Y%m%dT%H%M}"
    return [
        {
            "unique_id": spec.unique_id,
            "ds": span[i],
            "is_anomaly": True,
            "event_id": event_id,
            "anomaly_type": spec.kind,
            "severity": float(severity[i]) if np.isfinite(severity[i]) else None,
            "injection_seed": seed,
        }
        for i in range(len(span))
        if affected[i]
    ]
