"""Runs de backtesting sinteticos con residuos de distribucion controlada.

El detector conformal se mide por su tasa de marcado, y una tasa solo significa
algo si se conoce la distribucion que la genero. Estas fabricas construyen un
`BacktestResult` cuyas ventanas salen del splitter real y cuyos residuos salen de
una funcion que el test elige: homoscedastica, dependiente de la hora, a la
deriva o con un desplazamiento inyectado.

El `ScoringFrame` se sigue construyendo con `artifacts.reader.scoring_frame`, es
decir por el unico camino que existe (docs/ARCHITECTURE.md D12). Lo que estas
fabricas fingen es el run, no la barrera.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from chronolab.evaluation.backtest import BacktestPlan, BacktestResult
from chronolab.models.protocols import QUANTILES, quantile_column
from chronolab.panel import Panel, PanelSpec
from chronolab.types import ModelId
from tests.fixtures.synthetic import hourly_spec, make_hourly_panel

MODEL = ModelId("probe_conformal")
"""Modelo ficticio del que salen las predicciones de estos runs."""

Span = tuple[pd.Timestamp, pd.Timestamp]
"""Extremos del panel completo, para que un residuo pueda depender del tiempo absoluto."""

ResidualFn = Callable[[str, pd.DatetimeIndex, np.random.Generator, Span], np.ndarray]
"""Genera el residuo ``y - y_hat`` de una serie sobre un tramo de rejilla.

Recibe el tramo de la ventana en curso y, aparte, los extremos del panel entero:
sin ellos una varianza a la deriva se reiniciaria en cada ventana en lugar de
recorrer el run, que es justo el regimen que se quiere reproducir.
"""


def homoscedastic(sigma: float = 1.0) -> ResidualFn:
    """Residuos gaussianos de varianza constante.

    Parameters
    ----------
    sigma
        Desviacion tipica.

    Returns
    -------
    ResidualFn
        Funcion de residuos.
    """

    def residual(
        uid: str, ds: pd.DatetimeIndex, rng: np.random.Generator, span: Span
    ) -> np.ndarray:
        return rng.normal(0.0, sigma, len(ds))

    return residual


def hour_dependent(low: float = 0.4, high: float = 2.5) -> ResidualFn:
    """Residuos cuya varianza depende de la hora local, como la demanda electrica.

    Parameters
    ----------
    low, high
        Desviacion tipica de las horas tranquilas y de la punta.

    Returns
    -------
    ResidualFn
        Funcion de residuos. La punta cae en las horas 18-21 locales.
    """

    def residual(
        uid: str, ds: pd.DatetimeIndex, rng: np.random.Generator, span: Span
    ) -> np.ndarray:
        hour = ds.tz_localize("UTC").tz_convert("Europe/Madrid").hour.to_numpy()
        sigma = np.where((hour >= 18) & (hour <= 21), high, low)
        return rng.normal(0.0, 1.0, len(ds)) * sigma

    return residual


def drifting(start: float = 0.6, end: float = 3.0, onset: float = 0.0) -> ResidualFn:
    """Residuos cuya varianza crece linealmente a partir de cierto punto del panel.

    Con `onset` en la frontera entre desarrollo y holdout, la calibracion ve una
    distribucion y la puntuacion ve otra. Es el regimen en el que split conformal
    falla en silencio: la intercambiabilidad se rompe justo cuando importa, y la
    garantia de muestra finita sigue impresa en el informe.

    Parameters
    ----------
    start, end
        Desviacion tipica antes de `onset` y al final del panel.
    onset
        Fraccion del panel a partir de la cual empieza la deriva.

    Returns
    -------
    ResidualFn
        Funcion de residuos.
    """

    def residual(
        uid: str, ds: pd.DatetimeIndex, rng: np.random.Generator, span: Span
    ) -> np.ndarray:
        # Division de Timedelta y no aritmetica sobre enteros: `asi8` y
        # `Timestamp.value` no siempre estan en la misma unidad, y mezclarlos
        # deja la rampa plana sin que nada falle.
        total = max(span[1] - span[0], pd.Timedelta(1, "h"))
        position = np.asarray((ds - span[0]) / total, dtype=float)
        progress = np.clip((position - onset) / max(1.0 - onset, 1e-9), 0.0, 1.0)
        sigma = start + (end - start) * progress
        return rng.normal(0.0, 1.0, len(ds)) * sigma

    return residual


def make_result(
    *,
    residual: ResidualFn,
    n_series: int = 1,
    h: int = 24,
    n_windows: int = 60,
    holdout_windows: int = 10,
    gap: int = 0,
    half_width: float = 3.0,
    seed: int = 0,
    spec: PanelSpec | None = None,
) -> BacktestResult:
    """Run de backtesting teselado con residuos a medida.

    Las ventanas las emite el splitter real, de modo que el resultado cumple las
    mismas relaciones temporales que produciria el motor. Lo unico sintetico son
    ``y_hat`` y los cuantiles, que se derivan del residuo pedido.

    Parameters
    ----------
    residual
        Generador del residuo ``y - y_hat``.
    n_series
        Series del panel.
    h
        Horizonte. El plan es teselado, es decir ``step_size == h``.
    n_windows, holdout_windows
        Ventanas totales y ventanas finales reservadas para puntuar.
    gap
        Pasos entre el cutoff y la primera prediccion.
    half_width
        Semianchura del intervalo que declara el modelo. Constante a proposito:
        asi la heterocedasticidad del residuo llega entera al detector en lugar
        de quedar absorbida por el propio modelo.
    seed
        Semilla del generador.
    spec
        Especificacion del panel. Por defecto la horaria sintetica.

    Returns
    -------
    BacktestResult
        Con `forecasts`, `windows` y `model_runs` coherentes entre si.
    """
    used = spec if spec is not None else hourly_spec()
    n_hours = (n_windows + 1) * h + gap + 1
    panel = Panel(
        df=make_hourly_panel(n_series=n_series, n_hours=n_hours).df,
        spec=used,
    )
    plan = BacktestPlan(
        h=h,
        n_windows=n_windows,
        step_size=h,
        gap=gap,
        holdout_windows=holdout_windows,
    )
    windows = plan.splitter().split(panel)
    rng = np.random.default_rng(seed)
    span = (windows[0].first_pred, windows[-1].last_pred)

    parts: list[pd.DataFrame] = []
    for window in windows:
        actuals = panel.actuals(window)
        for uid, group in actuals.groupby("unique_id", sort=True):
            index = pd.DatetimeIndex(group["ds"])
            observed = group[used.target].to_numpy(dtype=float)
            error = residual(str(uid), index, rng, span)
            predicted = observed - error
            frame = pd.DataFrame(
                {
                    "unique_id": str(uid),
                    "model_id": str(MODEL),
                    "window_id": window.window_id,
                    "cutoff": window.cutoff,
                    "ds": index,
                    "h_step": np.arange(1, len(index) + 1),
                    "y": observed,
                    "y_hat": predicted,
                }
            )
            for quantile in QUANTILES:
                offset = _offset(quantile, half_width)
                frame[quantile_column(quantile)] = predicted + offset
            parts.append(frame)

    forecasts = _cast(pd.concat(parts, ignore_index=True))
    return BacktestResult(
        forecasts=forecasts.sort_values(["model_id", "unique_id", "window_id", "ds"]).reset_index(
            drop=True
        ),
        windows=_windows_frame(panel, windows),
        model_runs=pd.DataFrame(
            {
                "model_id": [str(MODEL)] * len(windows),
                "window_id": [window.window_id for window in windows],
                "status": ["ok"] * len(windows),
            }
        ),
        spec=used,
        plan=plan,
        futr_vintage=None,
    )


def inject_shift(
    result: BacktestResult,
    *,
    uid: str,
    start: pd.Timestamp,
    length: int,
    magnitude: float,
) -> BacktestResult:
    """Suma un desplazamiento de nivel a `y` en un tramo de una serie.

    Solo mueve la observacion: la prediccion y los cuantiles se quedan donde
    estaban, que es lo que hace que el tramo se salga del intervalo.

    Parameters
    ----------
    result
        Run al que inyectar.
    uid
        Serie afectada.
    start
        Primer instante del desplazamiento.
    length
        Longitud en pasos.
    magnitude
        Cuanto se suma a `y`.

    Returns
    -------
    BacktestResult
        Copia con la observacion desplazada.
    """
    forecasts = result.forecasts.copy()
    grid = pd.date_range(start, periods=length, freq=result.spec.freq)
    affected = (forecasts["unique_id"] == uid) & forecasts["ds"].isin(grid)
    forecasts.loc[affected, "y"] = forecasts.loc[affected, "y"] + np.float32(magnitude)
    return BacktestResult(
        forecasts=forecasts,
        windows=result.windows,
        model_runs=result.model_runs,
        spec=result.spec,
        plan=result.plan,
        futr_vintage=result.futr_vintage,
    )


def drop_window(
    result: BacktestResult, *, window_id: int, uid: str | None = None
) -> BacktestResult:
    """Borra las predicciones de una ventana, como si el modelo hubiese fallado alli.

    El motor registra ese fallo con ``status="failed"`` y no escribe filas, asi
    que el tramo queda con un hueco. Sirve para comprobar que el lector lo
    devuelve como `NaN` explicito y que el detector lo marca no puntuable en vez
    de desalinear la rejilla.

    Parameters
    ----------
    result
        Run al que quitarle una ventana.
    window_id
        Ventana afectada.
    uid
        Serie afectada. ``None`` borra todas.

    Returns
    -------
    BacktestResult
        Copia sin esas filas en `forecasts`.
    """
    forecasts = result.forecasts
    doomed = forecasts["window_id"] == window_id
    if uid is not None:
        doomed &= forecasts["unique_id"] == uid
    return BacktestResult(
        forecasts=forecasts.loc[~doomed].reset_index(drop=True),
        windows=result.windows,
        model_runs=result.model_runs,
        spec=result.spec,
        plan=result.plan,
        futr_vintage=result.futr_vintage,
    )


def _offset(quantile: float, half_width: float) -> float:
    """Desplazamiento de un cuantil respecto de la prediccion puntual.

    Parameters
    ----------
    quantile
        Cuantil de la rejilla.
    half_width
        Semianchura del intervalo al 95 %.

    Returns
    -------
    float
        Distancia con signo desde ``y_hat``, lineal en el cuantil para que la
        banda sea simetrica y facil de comprobar a mano.
    """
    return half_width * (quantile - 0.5) / (0.975 - 0.5)


def _cast(frame: pd.DataFrame) -> pd.DataFrame:
    """Fija los tipos de `forecasts` como los deja el motor.

    Parameters
    ----------
    frame
        Trama con las columnas correctas.

    Returns
    -------
    pandas.DataFrame
        Con los tipos de docs/ARCHITECTURE.md §7.4.
    """
    typed = frame.copy()
    typed["window_id"] = typed["window_id"].astype("int16")
    typed["h_step"] = typed["h_step"].astype("int16")
    for column in typed.columns:
        if column in ("y", "y_hat") or column.startswith("q_"):
            typed[column] = typed[column].astype("float32")
    typed["ds"] = typed["ds"].astype("datetime64[ns]")
    typed["cutoff"] = typed["cutoff"].astype("datetime64[ns]")
    return typed


def _windows_frame(panel: Panel, windows: Sequence[object]) -> pd.DataFrame:
    """Tabla `windows` del run sintetico.

    Parameters
    ----------
    panel
        Panel evaluado.
    windows
        Ventanas del plan.

    Returns
    -------
    pandas.DataFrame
        Una fila por ventana.
    """
    rows = [
        {
            "window_id": window.window_id,  # type: ignore[attr-defined]
            "stage": window.stage,  # type: ignore[attr-defined]
            "train_start": window.train_start,  # type: ignore[attr-defined]
            "cutoff": window.cutoff,  # type: ignore[attr-defined]
            "first_pred": window.first_pred,  # type: ignore[attr-defined]
            "last_pred": window.last_pred,  # type: ignore[attr-defined]
            "n_train_obs": len(panel.df),
            "n_series": len(panel.ids()),
        }
        for window in windows
    ]
    frame = pd.DataFrame(rows)
    frame["window_id"] = frame["window_id"].astype("int16")
    return frame
