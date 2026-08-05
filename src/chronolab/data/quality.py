"""Informe de calidad: huecos, duplicados, ceros frente a NaN, saltos de DST.

Alimenta la pagina Overview de la app y es la senal temprana del riesgo R3 de
docs/ARCHITECTURE.md. Las funciones de este modulo son deliberadamente tontas:
cuentan y marcan, no deciden que hacer con lo que encuentran. Esa decision
(imputar, descartar, avisar) vive en el modelo o en quien lea el informe.

Dos tramas distintas entran en juego, y no son intercambiables:

- La trama **cruda**, tal como la devuelve ``DataSource.fetch()``: puede tener
  huecos, duplicados y no estar en rejilla regular.
- La trama **alineada**, tras ``align.to_utc_naive`` + ``align.reindex_to_full_grid``:
  ya tiene rejilla completa (los huecos son ``NaN`` explicito) y esta libre de
  duplicados. Duplicados y cobertura cruda solo se pueden medir en la cruda;
  huecos y atipicos se miden en la alineada.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "coverage_report",
    "detect_duplicates",
    "detect_outliers",
    "detect_zeros",
    "dst_transition_report",
]


def coverage_report(
    raw_frame: pd.DataFrame,
    aligned_frame: pd.DataFrame,
    *,
    value_column: str = "y",
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """Resumen de calidad por serie: cobertura, huecos, duplicados, ceros, atipicos.

    Parameters
    ----------
    raw_frame
        Trama cruda, tal como la devuelve ``DataSource.fetch()``.
    aligned_frame
        La misma trama tras ``to_utc_naive`` y ``reindex_to_full_grid``: rejilla
        completa, sin duplicados.
    value_column
        Columna sobre la que se miden ceros y atipicos.
    z_threshold
        Umbral del z-score robusto (mediana y MAD) por encima del cual una
        observacion se marca como atipica. Ver `detect_outliers`.

    Returns
    -------
    pandas.DataFrame
        Una fila por ``unique_id`` con las columnas ``first_ds``, ``last_ds``,
        ``n_raw``, ``n_duplicated_pairs``, ``n_expected_grid``, ``n_aligned``,
        ``n_gaps``, ``coverage`` (``n_aligned - n_gaps`` sobre ``n_expected_grid``),
        ``n_zeros`` y ``n_outliers``.
    """
    raw_span = raw_frame.groupby("unique_id")["ds"].agg(first_ds="min", last_ds="max")

    # `n_duplicated_pairs` cuenta pares, no filas: `detect_duplicates` devuelve
    # todas las ocurrencias (para poder inspeccionarlas), pero un par
    # duplicado con dos ocurrencias debe contar 1, no 2. `keep="first"` marca
    # unicamente las ocurrencias "sobrantes", que es exactamente ese conteo.
    is_extra_occurrence = raw_frame.duplicated(subset=["unique_id", "ds"], keep="first")
    dup_counts = (
        raw_frame.loc[is_extra_occurrence].groupby("unique_id").size().rename("n_duplicated_pairs")
    )

    n_raw = raw_frame.groupby("unique_id").size().rename("n_raw")
    n_aligned = aligned_frame.groupby("unique_id").size().rename("n_aligned")
    n_gaps = (
        aligned_frame.assign(_missing=aligned_frame[value_column].isna())
        .groupby("unique_id")["_missing"]
        .sum()
        .rename("n_gaps")
        .astype(int)
    )

    zeros = detect_zeros(aligned_frame, value_column=value_column)
    n_zeros = zeros.groupby("unique_id").size().rename("n_zeros")

    outliers = detect_outliers(aligned_frame, value_column=value_column, z_threshold=z_threshold)
    n_outliers = outliers.groupby("unique_id").size().rename("n_outliers")

    report = (
        raw_span.join([n_raw, dup_counts, n_aligned, n_gaps, n_zeros, n_outliers])
        .fillna({"n_duplicated_pairs": 0, "n_zeros": 0, "n_outliers": 0})
        .reset_index()
    )
    report["n_duplicated_pairs"] = report["n_duplicated_pairs"].astype(int)
    report["n_zeros"] = report["n_zeros"].astype(int)
    report["n_outliers"] = report["n_outliers"].astype(int)

    # Tamano esperado de una rejilla regular entre el primer y el ultimo dato
    # crudo. Cerca de una transicion de DST, la trama cruda en hora local
    # difiere de esto en +-1 fila por dia de transicion: es un artefacto
    # esperado del reloj de pared, no un hueco real. `dst_transition_report`
    # lo distingue explicitamente; aqui solo se cuenta.
    freq_hours = _infer_hourly_step(aligned_frame)
    span_hours = (report["last_ds"] - report["first_ds"]) / pd.Timedelta(hours=1)
    report["n_expected_grid"] = (span_hours / freq_hours).round().astype(int) + 1
    report["coverage"] = (report["n_aligned"] - report["n_gaps"]) / report["n_expected_grid"]

    columns = [
        "unique_id",
        "first_ds",
        "last_ds",
        "n_raw",
        "n_duplicated_pairs",
        "n_expected_grid",
        "n_aligned",
        "n_gaps",
        "coverage",
        "n_zeros",
        "n_outliers",
    ]
    result: pd.DataFrame = report[columns]
    return result


def detect_duplicates(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Filas cuyo par ``(unique_id, ds)`` aparece mas de una vez en la trama cruda.

    Parameters
    ----------
    raw_frame
        Trama cruda, antes de `align.deduplicate`.

    Returns
    -------
    pandas.DataFrame
        Subconjunto de `raw_frame` que participa en al menos un duplicado,
        **todas** las ocurrencias incluidas (no solo la sobrante), para que se
        pueda inspeccionar qué valores discrepan.
    """
    is_dup = raw_frame.duplicated(subset=["unique_id", "ds"], keep=False)
    return raw_frame.loc[is_dup]


def detect_zeros(frame: pd.DataFrame, *, value_column: str = "y") -> pd.DataFrame:
    """Filas con el valor exactamente en cero.

    Un cero es sospechoso en una serie de consumo o demanda estrictamente
    positiva: puede ser un sensor congelado, un corte real, o un valor
    faltante que se codifico como cero en vez de `NaN`. Esta funcion no
    decide cual es el caso; solo lo senala.

    Parameters
    ----------
    frame
        Trama larga con la columna `value_column`.
    value_column
        Columna a inspeccionar.

    Returns
    -------
    pandas.DataFrame
        Subconjunto de `frame` con ``frame[value_column] == 0``.
    """
    return frame.loc[frame[value_column] == 0]


def detect_outliers(
    frame: pd.DataFrame, *, value_column: str = "y", z_threshold: float = 4.0
) -> pd.DataFrame:
    """Filas cuyo z-score robusto, calculado por serie, supera un umbral.

    Se usa la mediana y la desviacion absoluta mediana (MAD) en vez de la
    media y la desviacion tipica: son robustas frente a los propios atipicos
    que se busca detectar, mientras que la media y la desviacion tipica se
    desplazan hacia ellos.

    Parameters
    ----------
    frame
        Trama larga con la columna `value_column`. Puede contener `NaN`
        (huecos): se ignoran, nunca se marcan como atipicos.
    value_column
        Columna a inspeccionar.
    z_threshold
        Umbral del z-score robusto absoluto. 4.0 es deliberadamente
        conservador: esto es un filtro de cordura para el perfil de calidad,
        no un detector de anomalias (ese es `chronolab.anomaly`).

    Returns
    -------
    pandas.DataFrame
        Subconjunto de `frame` marcado como atipico, con una columna adicional
        ``robust_z`` con el z-score calculado.
    """
    # Constante que hace que MAD sea consistente con la desviacion tipica bajo
    # normalidad: MAD * 1.4826 ~= sigma para una gaussiana.
    mad_to_sigma = 1.4826

    def _robust_z(values: pd.Series) -> pd.Series:
        median = values.median()
        mad = (values - median).abs().median()
        if mad == 0:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (values - median) / (mad * mad_to_sigma)

    z_scores = frame.groupby("unique_id")[value_column].transform(_robust_z)
    flagged = frame.assign(robust_z=z_scores)
    return flagged.loc[flagged["robust_z"].abs() > z_threshold]


def dst_transition_report(
    raw_frame: pd.DataFrame,
    aligned_frame: pd.DataFrame,
    *,
    transitions: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Compara filas por dia local crudo frente a huecos/duplicados en la version alineada.

    Para cada fecha de transicion, cuenta cuantas filas tenia ese dia civil en
    la trama cruda (23 en un salto de primavera, 25 en un vuelco de otono, si
    la fuente reporta hora local) y verifica que, tras `to_utc_naive` y
    `reindex_to_full_grid`, no queden duplicados de `(unique_id, ds)` ni huecos
    adicionales alrededor de esa fecha. Es la evidencia, no la promesa, de que
    el pipeline maneja el cambio de hora correctamente.

    Parameters
    ----------
    raw_frame
        Trama cruda, con `ds` en hora local (antes de `to_utc_naive`).
    aligned_frame
        La misma trama tras `to_utc_naive` y `reindex_to_full_grid`.
    transitions
        Fechas (a medianoche) de las transiciones de DST a inspeccionar.

    Returns
    -------
    pandas.DataFrame
        Una fila por transicion, con ``transition``, ``n_rows_local_day``
        (filas crudas ese dia civil), ``n_duplicated_ds_aligned`` (duplicados
        de ``ds`` en la version alineada dentro de esa ventana) y
        ``n_gap_rows_aligned`` (filas con valor nulo en esa ventana).
    """
    value_columns = [c for c in aligned_frame.columns if c not in ("unique_id", "ds")]
    rows = []
    for transition in transitions:
        day_start = pd.Timestamp(transition).normalize()
        day_end = day_start + pd.Timedelta(days=1)

        raw_day = raw_frame[(raw_frame["ds"] >= day_start) & (raw_frame["ds"] < day_end)]

        window = aligned_frame[
            (aligned_frame["ds"] >= day_start - pd.Timedelta(hours=3))
            & (aligned_frame["ds"] < day_end + pd.Timedelta(hours=3))
        ]
        n_duplicated = int(window.duplicated(subset=["unique_id", "ds"]).sum())
        n_gap_rows = int(window[value_columns[0]].isna().sum()) if value_columns else 0

        rows.append(
            {
                "transition": day_start,
                "n_rows_local_day": len(raw_day),
                "n_duplicated_ds_aligned": n_duplicated,
                "n_gap_rows_aligned": n_gap_rows,
            }
        )

    return pd.DataFrame(rows)


def _infer_hourly_step(aligned_frame: pd.DataFrame) -> float:
    """Estima el paso de la rejilla, en horas, a partir de una serie ya alineada."""
    one_series = aligned_frame.loc[aligned_frame["unique_id"] == aligned_frame["unique_id"].iloc[0]]
    diffs = one_series["ds"].sort_values().diff().dropna()
    if diffs.empty:
        return 1.0
    step: float = diffs.median() / pd.Timedelta(hours=1)
    return step
