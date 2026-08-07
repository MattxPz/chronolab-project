"""Colapsa puntuaciones puntuales en eventos con extension, severidad y emparejamiento.

Tres decisiones gobiernan este modulo, y las tres cambian los numeros:

1. **Los eventos se fusionan con tolerancia declarada.** Una anomalia real casi
   nunca produce una tirada ininterrumpida de marcas: un solo punto que vuelve a
   entrar en la banda parte un evento en dos y destroza la precision a nivel de
   evento. `merge_gap` dice cuanto se tolera, y se registra.
2. **La severidad acumulada existe.** `peak_severity` sola no distingue una hora
   muy fuera de seis horas ligeramente fuera, que operativamente son incidentes
   distintos. `cum_severity` es el area fuera de la banda.
3. **El emparejamiento con la verdad es uno a uno.** Si un evento real se parte
   en cinco detectados, el resultado es un acierto y cuatro falsas alarmas, nunca
   cinco aciertos. Es lo que impide reproducir el vicio del *point-adjusted F1*,
   que la literatura de benchmarking describe como una metrica que infla
   resultados hasta hacer que el ruido parezca estado del arte.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from chronolab.types import DetectorId

__all__ = [
    "EVENT_COLUMNS",
    "MATCH_COLUMNS",
    "NO_TOLERANCE",
    "TRUTH_EVENT_COLUMNS",
    "aggregate_events",
    "match_events",
    "truth_events",
]

EVENT_COLUMNS: tuple[str, ...] = (
    "detector_id",
    "unique_id",
    "event_id",
    "alpha",
    "start_ds",
    "end_ds",
    "n_points",
    "duration_steps",
    "peak_score",
    "peak_severity",
    "cum_severity",
    "peak_ds",
    "direction",
)
"""Columnas de `anomaly_events` anteriores al emparejamiento."""

MATCH_COLUMNS: tuple[str, ...] = (*EVENT_COLUMNS, "matched_truth_event_id", "match_kind")
"""Columnas de `anomaly_events` ya emparejada con la verdad."""

TRUTH_EVENT_COLUMNS: tuple[str, ...] = (
    "unique_id",
    "event_id",
    "start_ds",
    "end_ds",
    "n_points",
    "anomaly_type",
)
"""Columnas de la vista por eventos de `anomaly_truth`."""

NO_TOLERANCE: pd.Timedelta = pd.Timedelta(0)
"""Tolerancia nula de emparejamiento: el detectado tiene que solapar con el real."""

_HIT = "hit"
_FALSE_ALARM = "false_alarm"
_MISSED = "missed"


def aggregate_events(
    scores: pd.DataFrame,
    *,
    detector_id: DetectorId,
    alpha: float,
    merge_gap: int = 2,
    min_points: int = 1,
) -> pd.DataFrame:
    """Agrupa los puntos marcados de un detector en eventos.

    Parameters
    ----------
    scores
        Salida completa de `FittedDetector.score`, con la rejilla entera y
        ordenada por ``(unique_id, ds)``. Tiene que ser la salida completa: las
        distancias entre puntos se cuentan por posicion de fila, asi que una
        trama ya filtrada mentiria sobre la duracion.
    detector_id
        Detector que produjo los scores.
    alpha
        Tasa de falsos positivos objetivo. Un punto esta marcado cuando
        ``score >= -log10(alpha)``.
    merge_gap
        Puntos no marcados que se toleran dentro de un mismo evento.
    min_points
        Puntos marcados minimos para que un evento cuente. El valor por defecto
        es ``1`` a proposito: con alfa pequeno, el pico de un solo punto es uno
        de los cinco tipos que inyecta `chronolab.anomaly.injection`, y filtrarlo
        seria filtrar el objetivo.

    Returns
    -------
    pandas.DataFrame
        Columnas `EVENT_COLUMNS`, ordenada por ``(unique_id, start_ds)``.

        ``n_points`` y ``duration_steps`` difieren exactamente cuando hubo
        fusion, y esa diferencia es un diagnostico: dice si el evento fue solido
        o intermitente.

        ``cum_severity`` se mide siempre al nivel de referencia del detector, no
        a este `alpha`. Es deliberado: anclada a un unico nivel declarado, la
        magnitud de dos eventos derivados con alfas distintos sigue siendo
        comparable. La consecuencia es que con ``alpha`` mayor que el de
        referencia algun punto puede contribuir en negativo.

    Raises
    ------
    ValueError
        Si faltan columnas obligatorias, si `alpha` cae fuera de ``(0, 1)`` o si
        `merge_gap` o `min_points` son negativos.
    """
    required = {"unique_id", "ds", "score", "scorable", "severity", "side"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"faltan columnas obligatorias en los scores: {sorted(missing)}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha fuera de (0, 1): {alpha}")
    if merge_gap < 0:
        raise ValueError(f"merge_gap debe ser >= 0: {merge_gap}")
    if min_points < 1:
        raise ValueError(f"min_points debe ser >= 1: {min_points}")

    threshold = -math.log10(alpha)
    ordered = scores.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []

    for uid, group in ordered.groupby("unique_id", sort=True):
        score = group["score"].to_numpy(dtype=float)
        severity = group["severity"].to_numpy(dtype=float)
        side = group["side"].to_numpy(dtype=int)
        ds = group["ds"].to_numpy(dtype="datetime64[ns]")
        usable = group["scorable"].to_numpy(dtype=bool)
        flagged = usable & np.isfinite(score) & (score >= threshold)

        for start, end in _merged_runs(flagged, merge_gap=merge_gap):
            marked = np.flatnonzero(flagged[start : end + 1]) + start
            if marked.size < min_points:
                continue
            rows.append(
                _event_row(
                    detector_id=detector_id,
                    uid=str(uid),
                    alpha=alpha,
                    ds=ds,
                    score=score,
                    severity=severity,
                    side=side,
                    start=start,
                    end=end,
                    marked=marked,
                )
            )

    if not rows:
        return _empty(EVENT_COLUMNS)
    frame = pd.DataFrame(rows)[list(EVENT_COLUMNS)]
    return frame.sort_values(["unique_id", "start_ds"]).reset_index(drop=True)


def _merged_runs(flagged: np.ndarray, *, merge_gap: int) -> list[tuple[int, int]]:
    """Tiradas maximas de marcas, fusionando las separadas por huecos cortos.

    Parameters
    ----------
    flagged
        Mascara booleana de puntos marcados, en orden temporal.
    merge_gap
        Puntos no marcados que se toleran entre dos tiradas.

    Returns
    -------
    list of tuple
        Pares ``(inicio, fin)`` de posiciones, ambos inclusive y ambos marcados.
    """
    marked = np.flatnonzero(flagged)
    if marked.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = previous = int(marked[0])
    for position in marked[1:]:
        current = int(position)
        if current - previous - 1 > merge_gap:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    return runs


def _event_row(
    *,
    detector_id: DetectorId,
    uid: str,
    alpha: float,
    ds: np.ndarray,
    score: np.ndarray,
    severity: np.ndarray,
    side: np.ndarray,
    start: int,
    end: int,
    marked: np.ndarray,
) -> dict[str, object]:
    """Construye la fila de un evento a partir de sus puntos marcados.

    Parameters
    ----------
    detector_id
        Detector que lo produjo.
    uid
        Serie.
    alpha
        Nivel con el que se derivo.
    ds, score, severity, side
        Vectores de la serie completa.
    start, end
        Extremos del evento, en posiciones.
    marked
        Posiciones marcadas dentro del evento.

    Returns
    -------
    dict
        Fila lista para `EVENT_COLUMNS`.
    """
    severities = severity[marked]
    finite = severities[np.isfinite(severities)]
    peak_position = int(marked[int(np.nanargmax(severities))]) if finite.size else int(marked[0])
    sides = set(side[marked].tolist())
    start_ds = pd.Timestamp(ds[start])

    return {
        "detector_id": str(detector_id),
        "unique_id": uid,
        "event_id": f"{detector_id}|{uid}|{alpha:.4f}|{start_ds:%Y%m%dT%H%M}",
        "alpha": alpha,
        "start_ds": start_ds,
        "end_ds": pd.Timestamp(ds[end]),
        "n_points": int(marked.size),
        "duration_steps": int(end - start + 1),
        "peak_score": float(np.nanmax(score[marked])),
        "peak_severity": float(finite.max()) if finite.size else math.nan,
        "cum_severity": float(finite.sum()) if finite.size else math.nan,
        "peak_ds": pd.Timestamp(ds[peak_position]),
        "direction": _direction(sides),
    }


def _direction(sides: set[int]) -> str:
    """Sentido de un evento a partir de los lados de sus puntos marcados.

    En demanda electrica un pico de consumo y una caida son incidentes
    operativamente distintos, y colapsarlos pierde el bit mas accionable.

    Parameters
    ----------
    sides
        Lados observados: ``+1`` por arriba, ``-1`` por abajo.

    Returns
    -------
    str
        ``"over"``, ``"under"`` o ``"mixed"``.
    """
    if sides == {1}:
        return "over"
    if sides == {-1}:
        return "under"
    return "mixed"


def truth_events(truth: pd.DataFrame) -> pd.DataFrame:
    """Vista por eventos de la tabla de verdad inyectada.

    Parameters
    ----------
    truth
        `anomaly_truth`: ``unique_id``, ``ds``, ``is_anomaly``, ``event_id`` y,
        opcionalmente, ``anomaly_type``.

    Returns
    -------
    pandas.DataFrame
        Columnas `TRUTH_EVENT_COLUMNS`, una fila por evento real.

    Raises
    ------
    ValueError
        Si faltan columnas obligatorias.
    """
    required = {"unique_id", "ds", "is_anomaly", "event_id"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"faltan columnas obligatorias en la verdad: {sorted(missing)}")

    marked = truth.loc[truth["is_anomaly"].astype(bool) & truth["event_id"].notna()]
    if marked.empty:
        return _empty(TRUTH_EVENT_COLUMNS)

    grouped = marked.groupby(["unique_id", "event_id"], sort=True)
    frame = grouped.agg(start_ds=("ds", "min"), end_ds=("ds", "max"), n_points=("ds", "size"))
    frame = frame.reset_index()
    if "anomaly_type" in marked.columns:
        types = grouped["anomaly_type"].first().reset_index()
        frame = frame.merge(types, on=["unique_id", "event_id"], how="left")
    else:
        frame["anomaly_type"] = None
    ordered = frame[list(TRUTH_EVENT_COLUMNS)].sort_values(["unique_id", "start_ds"])
    return ordered.reset_index(drop=True)


def match_events(
    detected: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    alpha: float,
    tol: pd.Timedelta = NO_TOLERANCE,
) -> pd.DataFrame:
    """Empareja eventos detectados con eventos reales, uno a uno.

    Un detectado empareja con un real si se solapan tras expandir el real `tol` a
    cada lado, que es la tolerancia de latencia de deteccion. La asignacion es
    voraz por solape descendente y **uno a uno**: un evento real no puede
    justificar dos detecciones. Sin esa restriccion, partir un evento real en
    trozos subiria el recall sin coste, que es exactamente como las metricas
    ajustadas por punto inflan resultados.

    Parameters
    ----------
    detected
        Salida de `aggregate_events` para un unico `alpha`.
    truth
        Salida de `truth_events`.
    alpha
        Nivel con el que se derivaron los eventos detectados. Se pide explicito
        porque las filas de tipo ``missed`` tambien lo llevan y pueden existir
        aunque no se haya detectado nada.
    tol
        Tolerancia temporal aplicada a cada lado del evento real.

    Returns
    -------
    pandas.DataFrame
        Columnas `MATCH_COLUMNS`. Los detectados sin pareja son
        ``false_alarm``; los reales sin pareja aparecen como ``missed``, con
        `event_id` nulo y solo `unique_id`, `alpha` y el identificador real
        rellenos.

    Raises
    ------
    ValueError
        Si `detected` mezcla varios `alpha`, o si no coincide con el pedido.
    """
    if not detected.empty:
        present = {float(value) for value in detected["alpha"].unique()}
        if present != {alpha}:
            raise ValueError(
                f"match_events empareja un unico alpha: se pidio {alpha} y los eventos "
                f"traen {sorted(present)}"
            )

    detected_ids = detected["unique_id"].astype(str).to_numpy()
    truth_ids = truth["unique_id"].astype(str).to_numpy()
    detected_spans = _spans(detected)
    truth_spans = _spans(truth)
    truth_event_ids = truth["event_id"].astype(str).tolist()

    matched = np.full(len(detected), None, dtype=object)
    kind = np.full(len(detected), _FALSE_ALARM, dtype=object)
    missed: list[dict[str, object]] = []

    for uid in sorted(set(detected_ids.tolist()) | set(truth_ids.tolist())):
        mine = np.flatnonzero(detected_ids == uid).tolist()
        theirs = np.flatnonzero(truth_ids == uid).tolist()
        pairing = _pair(detected_spans, truth_spans, mine, theirs, tol=tol)
        for position, other in pairing.items():
            matched[position] = truth_event_ids[other]
            kind[position] = _HIT
        for other in theirs:
            if other in pairing.values():
                continue
            start, end = truth_spans[other]
            row: dict[str, object] = dict.fromkeys(MATCH_COLUMNS)
            row["unique_id"] = uid
            row["alpha"] = alpha
            row["start_ds"] = start
            row["end_ds"] = end
            row["matched_truth_event_id"] = truth_event_ids[other]
            row["match_kind"] = _MISSED
            missed.append(row)

    frame = detected.copy()
    frame["matched_truth_event_id"] = matched
    frame["match_kind"] = kind
    if missed:
        frame = pd.concat([frame, pd.DataFrame(missed)], ignore_index=True)
    if frame.empty:
        return _empty(MATCH_COLUMNS)
    ordered = frame[list(MATCH_COLUMNS)].sort_values(["unique_id", "match_kind", "start_ds"])
    return ordered.reset_index(drop=True)


def _spans(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Extremos de cada evento como pareja de marcas de tiempo.

    Parameters
    ----------
    frame
        Eventos con `start_ds` y `end_ds`.

    Returns
    -------
    list of tuple
        Un par por fila, en el orden de la trama.
    """
    return [
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in zip(frame["start_ds"], frame["end_ds"], strict=True)
    ]


def _pair(
    detected: list[tuple[pd.Timestamp, pd.Timestamp]],
    truth: list[tuple[pd.Timestamp, pd.Timestamp]],
    mine: list[int],
    theirs: list[int],
    *,
    tol: pd.Timedelta,
) -> dict[int, int]:
    """Asignacion voraz uno a uno entre eventos detectados y reales de una serie.

    Se ordenan los pares candidatos por solape descendente y se asignan
    saltandose los que ya tienen pareja. Que sea uno a uno es la propiedad que
    importa: un evento real no puede justificar dos detecciones, asi que partir
    un evento real en trozos no sube el recall.

    Parameters
    ----------
    detected, truth
        Extremos de todos los eventos, detectados y reales.
    mine, theirs
        Posiciones de los eventos de la serie que se empareja.
    tol
        Tolerancia temporal aplicada a cada lado del evento real.

    Returns
    -------
    dict
        De posicion de detectado a posicion de real.
    """
    zero = pd.Timedelta(0)
    candidates: list[tuple[int, int, int]] = []
    for position in mine:
        start, end = detected[position]
        for other in theirs:
            real_start, real_end = truth[other]
            overlap = min(end, real_end + tol) - max(start, real_start - tol)
            if overlap >= zero:
                candidates.append((-overlap.value, position, other))

    candidates.sort()
    pairing: dict[int, int] = {}
    taken: set[int] = set()
    for _, position, other in candidates:
        if position in pairing or other in taken:
            continue
        pairing[position] = other
        taken.add(other)
    return pairing


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    """Trama vacia con las columnas dadas.

    Parameters
    ----------
    columns
        Nombres de columna.

    Returns
    -------
    pandas.DataFrame
        Sin filas.
    """
    return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
