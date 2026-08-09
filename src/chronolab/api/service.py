"""FastAPI de solo lectura sobre los artefactos de casi tiempo real.

``POST /forecast``, ``POST /anomalies``, ``GET /models`` y ``GET /health``.

Esta capa **no calcula nada**, por la misma razon que `chronolab.app`
(docs/ARCHITECTURE.md A5) y por la misma regla de dependencias (§2.1):
``api → artifacts.reader, config, types``. No importa `chronolab.models`,
`chronolab.evaluation` ni `chronolab.anomaly` -ni siquiera para construir el
cuerpo de la respuesta de ``/anomalies``, que en apariencia es "deteccion".
Todo lo que este modulo sirve ya lo calculo y lo dejo escrito
``scripts/refresh_data.py`` en `chronolab.config.live_dir()`; aqui solo se
lee, se filtra por los parametros de la peticion y se serializa. Un servidor
que recalculase en cada request arrastraria `torch`/`statsforecast` al
contenedor de la API y duplicaria, con otro camino de codigo, exactamente lo
que el refresco ya hizo con las barreras anti-fuga del motor de backtesting.

Consecuencia directa de "solo lectura": el `alpha` de ``POST /anomalies`` no
es libre. Cada refresco calibra el detector conformal a un unico `alpha` (ver
``scripts/refresh_data.py:ALPHA``) y colapsa los eventos a ese nivel; pedir
uno distinto no es "recalcular con otro umbral" -no hay con que- sino un
``422`` que dice cual es el vigente.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request

from chronolab.artifacts.reader import (
    available_live_artifacts,
    load_live_anomaly_events,
    load_live_forecasts,
    load_live_manifest,
)
from chronolab.config import live_dir
from chronolab.errors import ArtifactNotFound, ChronolabError

__all__ = ["app"]

STALE_AFTER = pd.Timedelta(hours=12)
"""Dos ciclos de refresco (`refresh-data.yml` corre cada seis horas). Pasado
esto sin un refresco nuevo, `/health` reporta `"stale"` en lugar de `"ok"`:
los datos siguen siendo los del ultimo refresco, pero ya no son "casi tiempo
real" bajo ningun criterio razonable."""

ALPHA_TOLERANCE = 1e-9
"""Margen de comparacion en punto flotante entre el `alpha` pedido y el
`alpha` con el que calibro el ultimo refresco."""

app = FastAPI(
    title="chronolab API",
    description=(
        "Lectura de casi tiempo real sobre los artefactos de "
        "`scripts/refresh_data.py`: predicciones con intervalos y eventos de "
        "anomalia del refresco mas reciente. No entrena ni recalcula nada "
        "(docs/ARCHITECTURE.md A5)."
    ),
    version="0.1.0",
)


def _get_live_dir() -> Path:
    """Proveedor de dependencia del directorio de casi tiempo real.

    Indireccion deliberada: los tests lo sustituyen con
    ``app.dependency_overrides[_get_live_dir]`` para apuntar a un directorio
    temporal sin tocar `chronolab.config.get_settings` ni variables de entorno.
    """
    return live_dir()


LiveDir = Annotated[Path, Depends(_get_live_dir)]


@app.exception_handler(ArtifactNotFound)
async def _artifact_not_found_handler(_request: Request, exc: ArtifactNotFound) -> JSONResponse:
    """Traduce la ausencia de un artefacto a `404` en vez de un `500` generico."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ChronolabError)
async def _chronolab_error_handler(_request: Request, exc: ChronolabError) -> JSONResponse:
    """Cualquier otro error propio del proyecto: `500` con el mensaje, no una traza cruda."""
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# --------------------------------------------------------------------------- #
# Modelos de peticion y respuesta
# --------------------------------------------------------------------------- #


class LiveArtifactStatus(BaseModel):
    """Que artefactos del ultimo refresco existen."""

    forecasts: bool
    windows: bool
    anomaly_scores: bool
    anomaly_events: bool


class HealthResponse(BaseModel):
    """Respuesta de `GET /health`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "checked_at": "2024-08-07T12:00:03Z",
                    "generated_at": "2024-08-07T06:00:00Z",
                    "age_seconds": 21603.0,
                    "run_id": "01J9ZQK8N3F5XW1R7T2Y6H4B0C",
                    "live_artifacts": {
                        "forecasts": True,
                        "windows": True,
                        "anomaly_scores": True,
                        "anomaly_events": True,
                    },
                }
            ]
        }
    )

    status: Literal["ok", "stale", "unavailable"] = Field(
        description=(
            "'ok': hay un refresco y tiene menos de 12h. 'stale': hay un "
            "refresco pero es mas viejo que eso. 'unavailable': el refresco "
            "no ha corrido nunca contra este `live_dir`."
        )
    )
    checked_at: datetime
    generated_at: datetime | None = Field(
        default=None, description="Inicio del ultimo refresco, o `None` si `status='unavailable'`."
    )
    age_seconds: float | None = None
    run_id: str | None = None
    live_artifacts: LiveArtifactStatus


class ModelSummary(BaseModel):
    """Un modelo con predicciones en el ultimo refresco."""

    model_id: str
    n_series: int
    n_rows: int
    last_cutoff: datetime


class ModelsResponse(BaseModel):
    """Respuesta de `GET /models`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "models": [
                        {
                            "model_id": "MSTL",
                            "n_series": 1,
                            "n_rows": 48,
                            "last_cutoff": "2024-08-07T05:00:00",
                        }
                    ]
                }
            ]
        }
    )

    models: list[ModelSummary]


class ForecastRequest(BaseModel):
    """Cuerpo de `POST /forecast`."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"unique_id": "ES", "horizon": 6, "models": None}]}
    )

    unique_id: str = Field(description="Serie a predecir, tal como aparece en el panel vivo.")
    horizon: int = Field(
        default=6,
        ge=1,
        description=(
            "Pasos hacia delante pedidos. Se recorta al horizonte del refresco "
            "(`h_step` maximo disponible) si se pide mas de lo que hay."
        ),
    )
    models: list[str] | None = Field(
        default=None,
        description="Modelos a incluir. `None` (por defecto) devuelve todos los disponibles.",
    )


class ForecastPoint(BaseModel):
    """Un paso predicho, con su intervalo."""

    ds: datetime
    h_step: int
    y_hat: float
    quantiles: dict[str, float] = Field(
        description="Cuantil (como texto, p. ej. '0.025') -> valor predicho en ese cuantil."
    )


class ModelForecast(BaseModel):
    """Prediccion de un modelo para la serie pedida."""

    model_id: str
    cutoff: datetime = Field(description="Ultimo instante conocido en el momento de predecir.")
    points: list[ForecastPoint]


class ForecastResponse(BaseModel):
    """Respuesta de `POST /forecast`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "unique_id": "ES",
                    "horizon": 6,
                    "models": [
                        {
                            "model_id": "MSTL",
                            "cutoff": "2024-08-07T05:00:00",
                            "points": [
                                {
                                    "ds": "2024-08-07T06:00:00",
                                    "h_step": 1,
                                    "y_hat": 28453.1,
                                    "quantiles": {
                                        "0.025": 27210.4,
                                        "0.5": 28453.1,
                                        "0.975": 29688.9,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    unique_id: str
    horizon: int
    models: list[ModelForecast]


class AnomalyRequest(BaseModel):
    """Cuerpo de `POST /anomalies`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "unique_id": "ES",
                    "start": "2024-08-06T00:00:00",
                    "end": "2024-08-07T06:00:00",
                    "alpha": 0.05,
                }
            ]
        }
    )

    unique_id: str
    start: datetime = Field(description="Extremo inicial (inclusive) del rango a consultar.")
    end: datetime = Field(description="Extremo final (inclusive) del rango a consultar.")
    alpha: float = Field(
        gt=0.0,
        lt=1.0,
        description=(
            "Tasa de falsos positivos objetivo. Debe coincidir (con tolerancia de punto "
            "flotante) con el `alpha` con el que calibro el ultimo refresco: no hay "
            "recalculo bajo demanda, ver el docstring del modulo."
        ),
    )


class AnomalyEvent(BaseModel):
    """Un evento de anomalia colapsado (`chronolab.anomaly.events.aggregate_events`)."""

    detector_id: str
    event_id: str
    start_ds: datetime
    end_ds: datetime
    n_points: int
    duration_steps: int
    peak_score: float
    peak_severity: float | None
    cum_severity: float | None
    peak_ds: datetime
    direction: str


class AnomalyResponse(BaseModel):
    """Respuesta de `POST /anomalies`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "unique_id": "ES",
                    "alpha": 0.05,
                    "start": "2024-08-06T00:00:00",
                    "end": "2024-08-07T06:00:00",
                    "events": [],
                }
            ]
        }
    )

    unique_id: str
    alpha: float
    start: datetime
    end: datetime
    events: list[AnomalyEvent]


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse)
def health(live_dir_path: LiveDir) -> HealthResponse:
    """Estado del ultimo refresco: si existe, su edad y que artefactos trae."""
    artifacts = LiveArtifactStatus(**available_live_artifacts(live_dir_path))
    try:
        manifest = load_live_manifest(live_dir=live_dir_path)
    except ArtifactNotFound:
        return HealthResponse(
            status="unavailable", checked_at=datetime.now(UTC), live_artifacts=artifacts
        )

    generated_at = datetime.fromisoformat(str(manifest["generated_at"]))
    checked_at = datetime.now(UTC)
    age = checked_at - generated_at
    status: Literal["ok", "stale"] = "ok" if age <= STALE_AFTER else "stale"
    return HealthResponse(
        status=status,
        checked_at=checked_at,
        generated_at=generated_at,
        age_seconds=age.total_seconds(),
        run_id=str(manifest.get("run_id")) if manifest.get("run_id") is not None else None,
        live_artifacts=artifacts,
    )


@app.get("/models", response_model=ModelsResponse)
def list_models(live_dir_path: LiveDir) -> ModelsResponse:
    """Modelos con predicciones en el ultimo refresco, y cuanto abarcan.

    Raises
    ------
    fastapi.HTTPException
        404 si el refresco no ha corrido todavia (via el manejador de
        `chronolab.errors.ArtifactNotFound`).
    """
    forecasts = load_live_forecasts(live_dir=live_dir_path)
    grouped = forecasts.groupby("model_id").agg(
        n_series=("unique_id", "nunique"),
        n_rows=("model_id", "size"),
        last_cutoff=("cutoff", "max"),
    )
    models = [
        ModelSummary(
            model_id=str(model_id),
            n_series=int(row["n_series"]),
            n_rows=int(row["n_rows"]),
            last_cutoff=pd.Timestamp(row["last_cutoff"]).to_pydatetime(),
        )
        for model_id, row in grouped.iterrows()
    ]
    return ModelsResponse(models=models)


def _quantile_label(column: str) -> str:
    """Convierte una columna ``q_0250`` en la etiqueta de cuantil ``"0.025"``.

    Parameters
    ----------
    column
        Nombre de columna con el convenio de docs/ARCHITECTURE.md §7.3.

    Returns
    -------
    str
        Cuantil como texto decimal.
    """
    return str(int(column.removeprefix("q_")) / 10_000)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest, live_dir_path: LiveDir) -> ForecastResponse:
    """Prediccion mas reciente para una serie, con intervalos, por modelo.

    Devuelve la ultima ventana disponible (el `cutoff` mas reciente) de cada
    modelo pedido, recortada a `request.horizon` pasos.

    Raises
    ------
    fastapi.HTTPException
        404 si no hay refresco o la serie no aparece en el; 404 tambien si se
        pide un `model_id` que no tiene predicciones para esa serie.
    """
    forecasts = load_live_forecasts(live_dir=live_dir_path)
    of_series = forecasts.loc[forecasts["unique_id"] == request.unique_id]
    if of_series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"la serie '{request.unique_id}' no aparece en el ultimo refresco",
        )

    requested_models: Sequence[str] = request.models or sorted(of_series["model_id"].unique())
    quantile_columns = [c for c in forecasts.columns if c.startswith("q_")]

    model_forecasts: list[ModelForecast] = []
    for model_id in requested_models:
        of_model = of_series.loc[of_series["model_id"] == model_id]
        if of_model.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"el modelo '{model_id}' no tiene predicciones para "
                    f"'{request.unique_id}' en el ultimo refresco"
                ),
            )
        latest_cutoff = of_model["cutoff"].max()
        latest = of_model.loc[
            (of_model["cutoff"] == latest_cutoff) & (of_model["h_step"] <= request.horizon)
        ].sort_values("h_step")

        points = [
            ForecastPoint(
                ds=pd.Timestamp(row["ds"]).to_pydatetime(),
                h_step=int(row["h_step"]),
                y_hat=float(row["y_hat"]),
                quantiles={
                    _quantile_label(col): float(row[col])
                    for col in quantile_columns
                    if pd.notna(row[col])
                },
            )
            for _, row in latest.iterrows()
        ]
        model_forecasts.append(
            ModelForecast(
                model_id=str(model_id),
                cutoff=pd.Timestamp(latest_cutoff).to_pydatetime(),
                points=points,
            )
        )

    return ForecastResponse(
        unique_id=request.unique_id, horizon=request.horizon, models=model_forecasts
    )


@app.post("/anomalies", response_model=AnomalyResponse)
def anomalies(request: AnomalyRequest, live_dir_path: LiveDir) -> AnomalyResponse:
    """Eventos de anomalia del ultimo refresco que solapan el rango pedido.

    El `alpha` de la peticion tiene que coincidir con el del ultimo refresco
    (ver el docstring del modulo): no hay umbral bajo demanda.

    Raises
    ------
    fastapi.HTTPException
        404 si no hay refresco todavia. 422 si `alpha` no coincide con el
        vigente -el mensaje incluye cual es-.
    """
    manifest = load_live_manifest(live_dir=live_dir_path)
    current_alpha = float(manifest["alpha"])
    if abs(request.alpha - current_alpha) > ALPHA_TOLERANCE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"el ultimo refresco calibro a alpha={current_alpha}, no se puede "
                f"servir alpha={request.alpha} sin recalcular (fuera del alcance de "
                "esta API de solo lectura)"
            ),
        )

    events = load_live_anomaly_events(live_dir=live_dir_path)
    of_series = events.loc[events["unique_id"] == request.unique_id]
    start = pd.Timestamp(request.start)
    end = pd.Timestamp(request.end)
    overlapping = of_series.loc[(of_series["start_ds"] <= end) & (of_series["end_ds"] >= start)]
    overlapping = overlapping.sort_values("start_ds")

    payload: list[AnomalyEvent] = [
        AnomalyEvent(
            detector_id=str(row["detector_id"]),
            event_id=str(row["event_id"]),
            start_ds=pd.Timestamp(row["start_ds"]).to_pydatetime(),
            end_ds=pd.Timestamp(row["end_ds"]).to_pydatetime(),
            n_points=int(row["n_points"]),
            duration_steps=int(row["duration_steps"]),
            peak_score=float(row["peak_score"]),
            peak_severity=float(row["peak_severity"]) if pd.notna(row["peak_severity"]) else None,
            cum_severity=float(row["cum_severity"]) if pd.notna(row["cum_severity"]) else None,
            peak_ds=pd.Timestamp(row["peak_ds"]).to_pydatetime(),
            direction=str(row["direction"]),
        )
        for _, row in overlapping.iterrows()
    ]
    return AnomalyResponse(
        unique_id=request.unique_id,
        alpha=current_alpha,
        start=request.start,
        end=request.end,
        events=payload,
    )
