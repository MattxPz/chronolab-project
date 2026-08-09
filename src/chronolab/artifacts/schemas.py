"""Esquemas pandera de cada tabla de artefacto y `SCHEMA_VERSION`.

El lector rechaza versiones desconocidas en lugar de interpretar mal columnas
ausentes.

Dos familias de tabla conviven en ``reports/results/`` y se tratan distinto:

- Las que produce un script de evaluacion externo a este paquete
  (``leaderboard``, ``anomaly_*``, ``tft_interpretability``): el esquema aqui
  comprueba que las columnas que la app necesita existen con el tipo
  correcto, pero no cierra la trama (``strict=False``). Un script de
  evaluacion que anade una columna nueva no debe romper la app.
- Las que genera ``scripts/build_demo_artifacts.py`` dentro de este mismo
  hito (``panel``, ``forecasts_demo``, ``windows_demo``, ``dm_matrix``,
  ``quality_*``, ``mstl_components``, ``difficulty``): el esquema es cerrado
  (``strict=True``) porque el productor y el consumidor son el mismo cambio y
  no hay excusa para que diverjan en silencio.

Ningun esquema de este modulo exige rejilla completa ni ausencia de huecos:
esa es la validacion de ``data.schemas``/``panel.py`` sobre el panel canonico,
que corre *antes* de que un artefacto llegue a disco. Aqui solo se comprueba
lo que la app necesita para no reventar al leer una columna ausente o de tipo
inesperado.
"""

from __future__ import annotations

import pandera.pandas as pa

__all__ = [
    "SCHEMA_VERSION",
    "anomaly_events_schema",
    "anomaly_results_schema",
    "anomaly_scores_schema",
    "anomaly_truth_schema",
    "difficulty_schema",
    "dm_matrix_schema",
    "forecasts_schema",
    "leaderboard_schema",
    "mstl_components_schema",
    "panel_schema",
    "quality_outliers_schema",
    "quality_report_schema",
    "tft_interpretability_schema",
    "windows_schema",
]

SCHEMA_VERSION = 1
"""Version de esquema de los artefactos que este codigo sabe leer.

No hay todavia un `manifest.json` que lo declare por run (`artifacts.writer`
sigue sin implementar): de momento es documental, y el sitio donde subirlo el
dia que un cambio de columnas rompa compatibilidad hacia atras.
"""

_UNIQUE_ID = pa.Column(str, nullable=False)
_DS = pa.Column("datetime64[ns]", nullable=False)
_FLOAT = pa.Column(float, nullable=True, coerce=True)
_FLOAT_REQUIRED = pa.Column(float, nullable=False, coerce=True)


def leaderboard_schema() -> pa.DataFrameSchema:
    """Metricas agregadas por modelo, serie (o el panel entero) y etapa.

    ``unique_id`` es anulable a proposito: la fila con ``unique_id=NaN`` es la
    marginalizacion sobre todas las series (`evaluation.aggregate`), no una
    serie ausente.
    """
    return pa.DataFrameSchema(
        {
            "model_id": pa.Column(str, nullable=False),
            "unique_id": pa.Column(str, nullable=True, coerce=True),
            "stage": pa.Column(str, nullable=False),
            "n_windows": pa.Column(int, nullable=False, coerce=True),
            "mae": _FLOAT,
            "rmse": _FLOAT,
            "mape": _FLOAT,
            "smape": _FLOAT,
            "mase": _FLOAT,
            "coverage_95": _FLOAT,
            "coverage_80": _FLOAT,
            "coverage_50": _FLOAT,
            "fit_seconds_total": _FLOAT,
            "predict_seconds_total": _FLOAT,
            "is_zero_shot": pa.Column(bool, nullable=False, coerce=True),
        },
        strict=False,
        coerce=True,
    )


def anomaly_scores_schema() -> pa.DataFrameSchema:
    """Scores crudos de los detectores sobre el tramo puntuado."""
    return pa.DataFrameSchema(
        {
            "unique_id": _UNIQUE_ID,
            "ds": _DS,
            "detector_id": pa.Column(str, nullable=False),
            "score": pa.Column(float, nullable=True, coerce=True),
            "scorable": pa.Column(bool, nullable=False, coerce=True),
            "severity": _FLOAT,
        },
        strict=False,
        coerce=True,
    )


def anomaly_results_schema() -> pa.DataFrameSchema:
    """Tabla larga de metricas de deteccion, en el formato de `evaluation.anomaly_metrics`."""
    return pa.DataFrameSchema(
        {
            "detector_id": pa.Column(str, nullable=False),
            "unique_id": pa.Column(str, nullable=True, coerce=True),
            "anomaly_type": pa.Column(str, nullable=False),
            "metric": pa.Column(str, nullable=False),
            "value": _FLOAT,
            "n_obs": pa.Column(int, nullable=False, coerce=True),
            "n_events": pa.Column(int, nullable=False, coerce=True),
        },
        strict=False,
        coerce=True,
    )


def anomaly_truth_schema() -> pa.DataFrameSchema:
    """Ground truth disperso de la inyeccion sintetica: solo instantes anomalos."""
    return pa.DataFrameSchema(
        {
            "unique_id": _UNIQUE_ID,
            "ds": pa.Column("datetime64[us]", nullable=False, coerce=True),
            "is_anomaly": pa.Column(bool, nullable=False, coerce=True),
            "event_id": pa.Column(str, nullable=False),
            "anomaly_type": pa.Column(str, nullable=False),
            "severity": _FLOAT,
        },
        strict=False,
        coerce=True,
    )


def anomaly_events_schema() -> pa.DataFrameSchema:
    """Eventos colapsados de `chronolab.anomaly.events.aggregate_events`.

    Mismas columnas que `chronolab.anomaly.events.EVENT_COLUMNS`. Lo produce
    ``scripts/refresh_data.py`` (o un script de evaluacion futuro) llamando a
    ``aggregate_events``; ``chronolab.api.service`` solo lo lee y filtra, sin
    importar `chronolab.anomaly` (docs/ARCHITECTURE.md §2.1: `api` no puede
    importar `anomaly`).
    """
    return pa.DataFrameSchema(
        {
            "detector_id": pa.Column(str, nullable=False),
            "unique_id": _UNIQUE_ID,
            "event_id": pa.Column(str, nullable=False),
            "alpha": _FLOAT_REQUIRED,
            "start_ds": _DS,
            "end_ds": _DS,
            "n_points": pa.Column(int, nullable=False, coerce=True),
            "duration_steps": pa.Column(int, nullable=False, coerce=True),
            "peak_score": _FLOAT_REQUIRED,
            "peak_severity": _FLOAT,
            "cum_severity": _FLOAT,
            "peak_ds": _DS,
            "direction": pa.Column(str, nullable=False),
        },
        strict=False,
        coerce=True,
    )


def tft_interpretability_schema() -> pa.DataFrameSchema:
    """Pesos de atencion (y, si existe, descomposicion) de un TFT."""
    return pa.DataFrameSchema(
        {
            "kind": pa.Column(str, nullable=False),
            "feature": pa.Column(str, nullable=True, coerce=True),
            "block": pa.Column(str, nullable=True, coerce=True),
            "value": _FLOAT,
        },
        strict=False,
        coerce=True,
    )


def panel_schema() -> pa.DataFrameSchema:
    """Panel demo persistido: la misma forma larga que `chronolab.panel.Panel.df`."""
    return pa.DataFrameSchema(
        {
            "unique_id": _UNIQUE_ID,
            "ds": _DS,
            "y": _FLOAT,
            "temp_c": _FLOAT,
            "voltage": _FLOAT,
        },
        strict=True,
        coerce=True,
    )


def quality_report_schema() -> pa.DataFrameSchema:
    """Salida de `chronolab.data.quality.coverage_report`, una fila por serie."""
    return pa.DataFrameSchema(
        {
            "unique_id": _UNIQUE_ID,
            "first_ds": _DS,
            "last_ds": _DS,
            "n_raw": pa.Column(int, nullable=False, coerce=True),
            "n_duplicated_pairs": pa.Column(int, nullable=False, coerce=True),
            "n_expected_grid": pa.Column(int, nullable=False, coerce=True),
            "n_aligned": pa.Column(int, nullable=False, coerce=True),
            "n_gaps": pa.Column(int, nullable=False, coerce=True),
            "coverage": _FLOAT_REQUIRED,
            "n_zeros": pa.Column(int, nullable=False, coerce=True),
            "n_outliers": pa.Column(int, nullable=False, coerce=True),
        },
        strict=True,
        coerce=True,
    )


def quality_outliers_schema() -> pa.DataFrameSchema:
    """Filas marcadas como atipicas por `chronolab.data.quality.detect_outliers`."""
    return pa.DataFrameSchema(
        {
            "unique_id": _UNIQUE_ID,
            "ds": _DS,
            "y": _FLOAT_REQUIRED,
            "robust_z": _FLOAT_REQUIRED,
        },
        strict=False,
        coerce=True,
    )


def mstl_components_schema() -> pa.DataFrameSchema:
    """Descomposicion MSTL precalculada por serie: observado, tendencia, estacionales y residuo."""
    return pa.DataFrameSchema(
        {
            "unique_id": _UNIQUE_ID,
            "ds": _DS,
            "observed": _FLOAT_REQUIRED,
            "trend": _FLOAT_REQUIRED,
            "seasonal_24": _FLOAT_REQUIRED,
            "seasonal_168": _FLOAT_REQUIRED,
            "resid": _FLOAT_REQUIRED,
        },
        strict=True,
        coerce=True,
    )


def difficulty_schema() -> pa.DataFrameSchema:
    """Salida de `chronolab.viz.plots.compute_difficulty_table`, una fila por serie."""
    return pa.DataFrameSchema(
        {
            "unique_id": _UNIQUE_ID,
            "trend_strength": _FLOAT_REQUIRED,
            "seasonal_strength_24": _FLOAT_REQUIRED,
            "seasonal_strength_168": _FLOAT_REQUIRED,
            "spectral_entropy": _FLOAT,
        },
        strict=True,
        coerce=True,
    )


def forecasts_schema() -> pa.DataFrameSchema:
    """Predicciones crudas del backtest demo: una fila por (modelo, serie, ventana, instante)."""
    return pa.DataFrameSchema(
        {
            "unique_id": _UNIQUE_ID,
            "model_id": pa.Column(str, nullable=False),
            "window_id": pa.Column(int, nullable=False, coerce=True),
            "stage": pa.Column(str, nullable=False),
            "cutoff": _DS,
            "ds": _DS,
            "h_step": pa.Column(int, nullable=False, coerce=True),
            "y": _FLOAT,
            "y_hat": _FLOAT_REQUIRED,
        },
        strict=False,  # las columnas q_* varian con la rejilla de cuantiles del plan
        coerce=True,
    )


def windows_schema() -> pa.DataFrameSchema:
    """Una fila por ventana efectiva del backtest demo."""
    return pa.DataFrameSchema(
        {
            "window_id": pa.Column(int, nullable=False, coerce=True),
            "stage": pa.Column(str, nullable=False),
            "train_start": _DS,
            "cutoff": _DS,
            "first_pred": _DS,
            "last_pred": _DS,
        },
        strict=False,
        coerce=True,
    )


def dm_matrix_schema() -> pa.DataFrameSchema:
    """Contrastes de Diebold-Mariano por pareja de modelos del backtest demo."""
    return pa.DataFrameSchema(
        {
            "model_a": pa.Column(str, nullable=False),
            "model_b": pa.Column(str, nullable=False),
            "stat": _FLOAT,
            "p_value": pa.Column(
                float, nullable=True, coerce=True, checks=pa.Check.in_range(0.0, 1.0)
            ),
            "n_obs": pa.Column(int, nullable=False, coerce=True),
            "mean_difference": _FLOAT,
            "degenerate": pa.Column(bool, nullable=False, coerce=True),
        },
        strict=True,
        coerce=True,
    )
