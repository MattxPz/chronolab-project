"""Refresco de casi tiempo real: datos nuevos, panel, prediccion y anomalias.

Es el unico script del repositorio pensado para correr desatendido y
repetidamente (`.github/workflows/refresh-data.yml` lo lanza cada seis horas),
no una vez por hito como `scripts/build_demo_artifacts.py` o
`scripts/run_anomaly_eval.py`. Eso impone dos propiedades que esos scripts no
necesitan:

- **Idempotente.** Cada paso es seguro de repetir: la descarga pasa por
  `chronolab.data.cache.CachedSource` (clave = fuente + rango + `as_of`, misma
  llamada siempre resuelve al mismo fichero), y la escritura final sobrescribe
  atomicamente el refresco anterior en lugar de acumular ficheros. Correr el
  script dos veces seguidas dentro de la misma hora no descarga nada dos
  veces ni deja el directorio de salida a medio escribir.
- **Con logging estructurado**, via `chronolab.logging`: cada linea es JSON,
  no prosa, porque nadie mira este log en una terminal en el momento en que se
  emite -lo mira una plataforma de observabilidad, o el propio `run_id` cuando
  algo falla tres refrescos despues.

Que hace, en orden:

1. **Descarga** demanda electrica de Espana (`REEDemandSource`, target) y
   temperatura de Madrid (`OpenMeteoSource`, exogena historica) de los
   ultimos `LOOKBACK_DAYS`, con la fuente real envuelta en `CachedSource` -eso
   es "la cache" que este script actualiza; no hay una segunda cache propia.
2. **Valida y ensambla** un `Panel`: cada fuente ya valida su propio esquema
   pandera dentro de `fetch()`; aqui se completa la rejilla horaria
   (invariante I3) y se construye `chronolab.data.quality.coverage_report`
   para que un tramo con demasiados huecos quede escrito y sea auditable, no
   silencioso (docs/ARCHITECTURE.md A6).
3. **Aplica el modelo configurado** (`MSTLForecaster`, `refit_cost` barato
   frente a un modelo neuronal) a los datos recientes: no hay ningun
   mecanismo de serializar-y-cargar un modelo ajustado en el resto del
   repositorio (ni un `.pkl`, ni un `models.registry` -sigue siendo un stub),
   asi que "modelo guardado" aqui significa la configuracion versionada en
   este fichero, reajustada en cada ventana del backtest teselado, que es
   exactamente como el motor de `chronolab.evaluation.backtest` trata a
   cualquier modelo de coste barato. Reajustar una MSTL sobre unos cientos de
   puntos cuesta segundos, no minutos: no hay nada que amortizar guardando
   estado entre refrescos.
4. **Detecta anomalias en modo adaptativo**: `ConformalDetector` calibra
   sobre el tramo `dev` del propio backtest y puntua el tramo `holdout` (las
   horas mas recientes) con inferencia conformal adaptativa (ACI, parametro
   `gamma`) -es el detector del proyecto que ya se adapta en linea, y "modo
   adaptativo" es exactamente su comportamiento por defecto, no un modo aparte
   que haya que activar.
5. **Escribe los artefactos actualizados** en `chronolab.config.live_dir()`
   (``data/artifacts/live/`` por defecto: gitignorado, no el subconjunto demo
   versionado de ``reports/results/``), y dentro de eso el manifest el
   ultimo, mismo convenio de atomicidad que docs/ARCHITECTURE.md §7.2 describe
   para un run completo.

Uso: ``uv run --extra ml python scripts/refresh_data.py``

Variables de entorno (todas opcionales, prefijo `CHRONOLAB_REFRESH_`):

- ``CHRONOLAB_REFRESH_LOOKBACK_DAYS`` (por defecto 45)
- ``CHRONOLAB_REFRESH_HORIZON`` (por defecto 6, pasos horarios por ciclo)
- ``CHRONOLAB_REFRESH_ALPHA`` (por defecto 0.05)
- ``CHRONOLAB_REFRESH_LOG_LEVEL`` (por defecto INFO)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from ulid import ULID  # noqa: E402

from chronolab.anomaly.conformal import ConformalDetector  # noqa: E402
from chronolab.anomaly.events import aggregate_events  # noqa: E402
from chronolab.artifacts import schemas  # noqa: E402
from chronolab.artifacts.reader import scoring_frame  # noqa: E402
from chronolab.config import live_dir  # noqa: E402
from chronolab.data.align import deduplicate, reindex_to_full_grid  # noqa: E402
from chronolab.data.cache import CachedSource  # noqa: E402
from chronolab.data.quality import coverage_report  # noqa: E402
from chronolab.data.sources.open_meteo import OpenMeteoSource  # noqa: E402
from chronolab.data.sources.ree import REEDemandSource  # noqa: E402
from chronolab.errors import SourceUnavailable, StaleCacheWarning  # noqa: E402
from chronolab.evaluation.backtest import BacktestPlan, BacktestResult, backtest  # noqa: E402
from chronolab.logging import bind_run_id, configure_logging, get_logger  # noqa: E402
from chronolab.models.adapters.statsforecast import MSTLForecaster  # noqa: E402
from chronolab.panel import Panel, PanelSpec  # noqa: E402
from chronolab.types import DatasetId  # noqa: E402

LOG = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #

DATASET_ID = DatasetId("es_demand_live")
UNIQUE_ID = "ES"
"""Unica serie del panel: demanda peninsular espanola. `OpenMeteoSource` trae su
propio `unique_id` derivado de la coordenada; se reasigna a este mismo id
porque, en este panel, demanda y temperatura describen una sola serie con dos
columnas, no dos series distintas."""

MADRID_LATITUDE = 40.4168
MADRID_LONGITUDE = -3.7038

LOOKBACK_DAYS = int(os.environ.get("CHRONOLAB_REFRESH_LOOKBACK_DAYS", "45"))
"""Historia descargada en cada refresco. Suficiente para `TRAIN_SIZE` mas
`DEV_WINDOWS + HOLDOUT_WINDOWS` ventanas con margen para huecos."""

PUBLICATION_LAG = pd.Timedelta(hours=3)
"""Las ultimas horas de demanda en tiempo real suelen revisarse tras su
publicacion inicial; se recorta la cola para no entrenar ni puntuar sobre un
valor que todavia puede cambiar."""

H = int(os.environ.get("CHRONOLAB_REFRESH_HORIZON", "6"))
"""Pasos por ventana. Igual al periodo del cron (seis horas): cada refresco
publica la prediccion y el estado de anomalias hasta el proximo."""

SEASON_LENGTH = 24
"""Solo estacionalidad diaria: `LOOKBACK_DAYS` no da ciclos semanales limpios
de sobra para calibrar el componente de 168 pasos ademas del de 24."""

TRAIN_SIZE = 24 * 14
"""Dos semanas de entrenamiento deslizante por ventana."""

DEV_WINDOWS = 40
HOLDOUT_WINDOWS = 4
"""Cuatro ventanas de seis pasos = 24 h puntuadas: el tramo que describe el
estado "ahora mismo" de la serie."""

ALPHA = float(os.environ.get("CHRONOLAB_REFRESH_ALPHA", "0.05"))
"""Nivel al que se binariza el score conformal y se colapsan los eventos."""

MERGE_GAP = 2
SEED = 20_240_807

CACHE_DIR = ROOT / "data" / "raw"
CACHE_MAX_AGE = pd.Timedelta(hours=1)
"""Una entrada de cache mas fresca que esto se sirve sin volver a consultar la
fuente: dos refrescos dentro de la misma hora no duplican trafico de red."""


@dataclass
class SourceOutcome:
    """Resultado de intentar poblar una columna de valor desde una fuente.

    Attributes
    ----------
    source_id
        Identificador de la fuente (`chronolab.data.protocols.SourceSpec.source_id`).
    status
        ``"ok"``, ``"stale_cache"`` (sirvio una entrada caducada porque la
        fuente no respondio) o ``"failed"`` (sin datos, ni frescos ni en cache).
    n_rows
        Filas obtenidas. ``0`` si `status` es ``"failed"``.
    """

    source_id: str
    status: str
    n_rows: int = 0


def _fetch_with_status(
    source: CachedSource, *, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, SourceOutcome]:
    """Descarga una fuente cacheada, degradando en lugar de propagar el fallo.

    Parameters
    ----------
    source
        Fuente ya envuelta en `CachedSource`.
    start, end
        Rango semiabierto `[start, end)` en UTC ingenuo.

    Returns
    -------
    tuple
        La trama obtenida (vacia si `status == "failed"`) y el resultado.

    Raises
    ------
    SourceUnavailable
        Nunca la propaga esta funcion: la convierte en `status="failed"` para
        que el llamante decida si esa fuente es imprescindible.
    """
    source_id = source.spec.source_id
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", StaleCacheWarning)
        try:
            frame = source.fetch(start=start, end=end)
        except SourceUnavailable as exc:
            LOG.error(
                "fuente no disponible y sin cache que servir",
                extra={"extra_fields": {"source_id": source_id, "error": str(exc)}},
            )
            return pd.DataFrame(), SourceOutcome(source_id=source_id, status="failed")

    stale = any(issubclass(w.category, StaleCacheWarning) for w in caught)
    status = "stale_cache" if stale else "ok"
    if stale:
        LOG.warning(
            "sirviendo cache obsoleta: la fuente no respondio",
            extra={"extra_fields": {"source_id": source_id}},
        )
    LOG.info(
        "fuente descargada",
        extra={"extra_fields": {"source_id": source_id, "status": status, "n_rows": len(frame)}},
    )
    return frame, SourceOutcome(source_id=source_id, status=status, n_rows=len(frame))


def fetch_recent_data(
    *, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, SourceOutcome]]:
    """Descarga demanda y temperatura recientes, cada una a traves de su cache.

    Parameters
    ----------
    end
        Extremo exclusivo del rango a descargar (ya recortado por `PUBLICATION_LAG`).

    Returns
    -------
    tuple
        Trama de demanda (``unique_id``, ``ds``, ``y``), trama de temperatura
        (``unique_id``, ``ds``, ``temp_c``) -vacia si la fuente fallo- y el
        resultado de cada fuente, por `source_id`.

    Raises
    ------
    SourceUnavailable
        Si la fuente de demanda -el objetivo, imprescindible- no tiene datos
        frescos ni en cache. La de temperatura es prescindible: sin ella el
        panel se construye sin exogena.
    """
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    demand_source = CachedSource(
        inner=REEDemandSource(), cache_dir=CACHE_DIR, max_age=CACHE_MAX_AGE
    )
    demand_frame, demand_outcome = _fetch_with_status(demand_source, start=start, end=end)
    if demand_outcome.status == "failed":
        raise SourceUnavailable(
            f"{demand_source.spec.source_id}: sin datos frescos ni en cache, no se puede refrescar"
        )

    weather_source = CachedSource(
        inner=OpenMeteoSource(latitude=MADRID_LATITUDE, longitude=MADRID_LONGITUDE),
        cache_dir=CACHE_DIR,
        max_age=CACHE_MAX_AGE,
    )
    weather_frame, weather_outcome = _fetch_with_status(weather_source, start=start, end=end)
    if not weather_frame.empty:
        weather_frame = weather_frame.assign(unique_id=UNIQUE_ID)

    outcomes = {
        demand_outcome.source_id: demand_outcome,
        weather_outcome.source_id: weather_outcome,
    }
    return demand_frame, weather_frame, outcomes


def build_live_panel(
    demand_frame: pd.DataFrame, weather_frame: pd.DataFrame
) -> tuple[Panel, pd.DataFrame]:
    """Ensambla demanda y temperatura en un `Panel` de una sola serie.

    No pasa por ``chronolab.data.assemble.build_panel`` (sin implementar
    todavia): construye el `Panel` directamente, con el mismo patron que usa
    `tests.fixtures.synthetic.make_hourly_panel` para las series sinteticas de
    la suite. `PanelSpec` sigue siendo la unica declaracion de roles: la
    temperatura entra como `hist_exog`, no `futr_exog`, porque ninguno de los
    modelos de este script la usa -sin ella no hace falta un `FutrProvider`
    ni resolver el vintage de la prevision meteorologica (docs/ARCHITECTURE.md
    §4.3), y queda igualmente disponible en el panel para inspeccion y para
    un modelo futuro que si la use.

    Parameters
    ----------
    demand_frame
        Trama cruda de `REEDemandSource`.
    weather_frame
        Trama cruda de `OpenMeteoSource`, ya reasignada al `unique_id` de la
        demanda. Puede estar vacia si la fuente fallo.

    Returns
    -------
    tuple
        `Panel` con rejilla horaria completa (invariante I3) y la trama cruda
        combinada (pre-rejilla), para `chronolab.data.quality.coverage_report`.
    """
    has_weather = not weather_frame.empty
    if has_weather:
        raw = demand_frame.merge(
            weather_frame[["unique_id", "ds", "temp_c"]], on=["unique_id", "ds"], how="left"
        )
    else:
        raw = demand_frame.assign(temp_c=pd.Series(dtype="float64"))

    raw = deduplicate(raw, policy="last")
    aligned = reindex_to_full_grid(raw, freq="h")
    aligned["y"] = aligned["y"].astype("float32")
    aligned["temp_c"] = aligned["temp_c"].astype("float32")

    spec = PanelSpec(
        dataset_id=DATASET_ID,
        freq="h",
        seasonalities=(SEASON_LENGTH,),
        hist_exog=("temp_c",),
        tz_display="Europe/Madrid",
    )
    panel = Panel(df=aligned[list(spec.columns)], spec=spec)
    return panel, raw


def build_plan() -> BacktestPlan:
    """Plan teselado con reajuste por ventana, requisito de `scoring_frame`."""
    return BacktestPlan(
        h=H,
        n_windows=DEV_WINDOWS + HOLDOUT_WINDOWS,
        step_size=H,
        gap=0,
        mode="sliding",
        train_size=TRAIN_SIZE,
        holdout_windows=HOLDOUT_WINDOWS,
        refit_every=1,
        seed=SEED,
    )


def run_forecast_and_detect(
    panel: Panel,
) -> tuple[BacktestResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest teselado + deteccion conformal adaptativa sobre el panel vivo.

    Parameters
    ----------
    panel
        Panel horario reciente.

    Returns
    -------
    tuple
        Resultado del backtest, la tabla `forecasts` con `stage` ya unido, los
        scores del detector conformal (rejilla completa del holdout) y los
        eventos colapsados a `ALPHA`.
    """
    model = MSTLForecaster(h=H, season_lengths=(SEASON_LENGTH,))
    plan = build_plan()

    started = time.perf_counter()
    with warnings.catch_warnings():
        # AutoARIMA (dentro de MSTL) avisa de convergencia en algunas ventanas
        # cortas; el ajuste que devuelve sigue siendo utilizable, igual que en
        # scripts/build_demo_artifacts.py y scripts/run_anomaly_eval.py.
        warnings.simplefilter("ignore", UserWarning)
        result = backtest(panel, [model], plan)
    elapsed = time.perf_counter() - started

    runs = result.model_runs
    failed = int((runs["status"] != "ok").sum())
    LOG.info(
        "backtest de refresco completado",
        extra={
            "extra_fields": {
                "model_id": str(model.model_id),
                "elapsed_seconds": round(elapsed, 2),
                "n_windows": len(runs),
                "n_failed": failed,
            }
        },
    )

    forecasts = result.forecasts.merge(
        result.windows[["window_id", "stage"]], on="window_id", how="left"
    )

    calib = scoring_frame(result, model_id=model.model_id, stage="dev")
    holdout = scoring_frame(result, model_id=model.model_id, stage="holdout")

    detector = ConformalDetector(base_model_id=model.model_id, alpha_nominal=ALPHA, alpha_ref=ALPHA)
    fitted = detector.fit(calib)
    scores = fitted.score(holdout)
    scores = scores.assign(detector_id=str(detector.detector_id))

    events = aggregate_events(
        scores, detector_id=detector.detector_id, alpha=ALPHA, merge_gap=MERGE_GAP
    )
    LOG.info(
        "deteccion de anomalias completada",
        extra={
            "extra_fields": {
                "detector_id": str(detector.detector_id),
                "n_scored": int(scores["scorable"].sum()),
                "n_events": len(events),
            }
        },
    )
    return result, forecasts, scores, events


# --------------------------------------------------------------------------- #
# Escritura atomica: fichero temporal + renombrado, manifest el ultimo
# --------------------------------------------------------------------------- #


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Escribe `frame` en `path` de forma atomica (mismo patron que `CachedSource._write`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Escribe `payload` como JSON en `path` de forma atomica."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    tmp_path.replace(path)


def _git_sha() -> str | None:
    """SHA del commit que genero el refresco, o `None` si no se puede resolver.

    Prefiere `GITHUB_SHA` (lo fija `refresh-data.yml` sin coste de proceso) y
    cae a `git rev-parse HEAD` para invocaciones locales.
    """
    from_env = os.environ.get("GITHUB_SHA")
    if from_env:
        return from_env
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def write_artifacts(
    *,
    panel: Panel,
    raw_frame: pd.DataFrame,
    result: BacktestResult,
    forecasts: pd.DataFrame,
    scores: pd.DataFrame,
    events: pd.DataFrame,
    outcomes: dict[str, SourceOutcome],
    run_id: str,
    started_at: datetime,
    elapsed_seconds: float,
    output_dir: Path,
) -> None:
    """Guarda el refresco en `output_dir`, con el manifest escrito el ultimo.

    Parameters
    ----------
    panel, raw_frame
        Panel alineado y su trama cruda previa a la rejilla, para el informe
        de calidad.
    result, forecasts, scores, events
        Salidas de `run_forecast_and_detect`.
    outcomes
        Resultado de cada fuente de datos (`fetch_recent_data`).
    run_id
        Identificador ULID de este refresco.
    started_at
        Instante UTC de inicio, para el manifest.
    elapsed_seconds
        Duracion total del refresco.
    output_dir
        Directorio de salida. `chronolab.config.live_dir()` por defecto.
    """
    quality = coverage_report(raw_frame, panel.df, value_column=panel.spec.target)

    _write_parquet_atomic(forecasts, output_dir / "forecasts.parquet")
    _write_parquet_atomic(result.windows, output_dir / "windows.parquet")
    _write_parquet_atomic(scores, output_dir / "anomaly_scores.parquet")
    _write_parquet_atomic(events, output_dir / "anomaly_events.parquet")
    _write_parquet_atomic(panel.df, output_dir / "panel.parquet")
    _write_parquet_atomic(quality, output_dir / "quality_report.parquet")

    coverage = float(quality["coverage"].iloc[0]) if len(quality) else 0.0
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": schemas.SCHEMA_VERSION,
        "generated_at": started_at.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "git_sha": _git_sha(),
        "git_dirty": None,
        "dataset_id": str(panel.spec.dataset_id),
        "freq": panel.spec.freq,
        "horizon": H,
        "alpha": ALPHA,
        "lookback_days": LOOKBACK_DAYS,
        "panel": {
            "n_rows": len(panel.df),
            "first_ds": panel.first_ds.isoformat(),
            "last_ds": panel.last_ds.isoformat(),
            "coverage": coverage,
        },
        "sources": {
            source_id: {"status": outcome.status, "n_rows": outcome.n_rows}
            for source_id, outcome in outcomes.items()
        },
        "backtest": {
            "model_id": str(forecasts["model_id"].iloc[0]) if len(forecasts) else None,
            "n_windows": len(result.windows),
            "n_forecast_rows": len(forecasts),
        },
        "anomaly": {
            "n_scored": int(scores["scorable"].sum()),
            "n_events": len(events),
        },
    }
    _write_json_atomic(manifest, output_dir / "manifest.json")


def main() -> None:
    """Ejecuta un ciclo completo de refresco."""
    log_level = getattr(logging, os.environ.get("CHRONOLAB_REFRESH_LOG_LEVEL", "INFO").upper())
    configure_logging(level=log_level)

    run_id = str(ULID())
    with bind_run_id(run_id):
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        LOG.info(
            "refresco iniciado",
            extra={
                "extra_fields": {
                    "lookback_days": LOOKBACK_DAYS,
                    "horizon": H,
                    "alpha": ALPHA,
                }
            },
        )

        end = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h") - PUBLICATION_LAG
        demand_frame, weather_frame, outcomes = fetch_recent_data(end=end)
        panel, raw_frame = build_live_panel(demand_frame, weather_frame)
        result, forecasts, scores, events = run_forecast_and_detect(panel)

        output_dir = live_dir()
        elapsed = time.perf_counter() - started
        write_artifacts(
            panel=panel,
            raw_frame=raw_frame,
            result=result,
            forecasts=forecasts,
            scores=scores,
            events=events,
            outcomes=outcomes,
            run_id=run_id,
            started_at=started_at,
            elapsed_seconds=elapsed,
            output_dir=output_dir,
        )
        LOG.info(
            "refresco completado",
            extra={
                "extra_fields": {
                    "elapsed_seconds": round(elapsed, 2),
                    "output_dir": str(output_dir),
                }
            },
        )


if __name__ == "__main__":
    main()
