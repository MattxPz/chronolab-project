"""Generador determinista de paneles horarios sinteticos para los tests.

Produce series con estacionalidad diaria (24) y semanal (168) conocidas, en UTC
ingenuo y con rejilla completa, de modo que los tests puedan verificar que el
arnes mide lo que dice medir antes de que los datos reales metan ruido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId

DAILY = 24
"""Longitud de la estacionalidad diaria en pasos horarios."""

WEEKLY = 168
"""Longitud de la estacionalidad semanal en pasos horarios."""

DEFAULT_START = "2023-01-02"
"""Lunes, para que el ciclo semanal empiece alineado con el inicio del panel."""


def hourly_spec(dataset_id: str = "synthetic_h") -> PanelSpec:
    """Especificacion del panel horario sintetico.

    Parameters
    ----------
    dataset_id
        Identificador del dataset.

    Returns
    -------
    PanelSpec
        Con estacionalidades ``(24, 168)``, ``temp_c`` como exogena futura y
        ``voltage`` como exogena historica.
    """
    return PanelSpec(
        dataset_id=DatasetId(dataset_id),
        freq="h",
        seasonalities=(DAILY, WEEKLY),
        futr_exog=("temp_c",),
        hist_exog=("voltage",),
        tz_display="Europe/Madrid",
    )


def make_hourly_frame(
    *,
    n_series: int = 3,
    n_hours: int = WEEKLY * 12,
    start: str = DEFAULT_START,
    seed: int = 0,
    daily_amp: float = 10.0,
    weekly_amp: float = 4.0,
    trend: float = 0.002,
    noise: float = 0.8,
) -> pd.DataFrame:
    """Genera una trama larga horaria con estacionalidad diaria y semanal.

    Parameters
    ----------
    n_series
        Numero de series del panel.
    n_hours
        Longitud de cada serie en horas.
    start
        Primera marca de tiempo, en UTC ingenuo.
    seed
        Semilla del generador. Fija el resultado por completo.
    daily_amp, weekly_amp
        Amplitud de las componentes de 24 y 168 pasos.
    trend
        Pendiente lineal por paso.
    noise
        Desviacion tipica del ruido gaussiano.

    Returns
    -------
    pandas.DataFrame
        Columnas ``unique_id``, ``ds``, ``y``, ``temp_c`` y ``voltage``, ordenada
        por ``(unique_id, ds)``, con rejilla completa y ``ds`` en UTC ingenuo.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=n_hours, freq="h")
    hour = np.asarray(index.hour, dtype=np.float64)
    hour_of_week = np.asarray(index.dayofweek * 24 + index.hour, dtype=np.float64)
    step = np.arange(n_hours, dtype=np.float64)

    daily = daily_amp * np.sin(2 * np.pi * hour / DAILY)
    weekly = weekly_amp * np.sin(2 * np.pi * hour_of_week / WEEKLY)

    parts: list[pd.DataFrame] = []
    for i in range(n_series):
        level = 100.0 + 10.0 * i
        temp_c = 12.0 + 8.0 * np.sin(2 * np.pi * (hour - 4.0) / DAILY) + rng.normal(0, 0.5, n_hours)
        # Relacion en "U" entre temperatura y demanda: sube con frio y con calor.
        thermal = 0.4 * np.abs(temp_c - 16.0)
        y = level + daily + weekly + trend * step + thermal + rng.normal(0, noise, n_hours)
        parts.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i:02d}",
                    "ds": index,
                    "y": y.astype(np.float32),
                    "temp_c": temp_c.astype(np.float32),
                    "voltage": rng.normal(230.0, 1.5, n_hours).astype(np.float32),
                }
            )
        )

    frame = pd.concat(parts, ignore_index=True)
    return frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def make_hourly_panel(**kwargs: object) -> Panel:
    """Panel horario sintetico listo para usar.

    Parameters
    ----------
    **kwargs
        Se reenvian a `make_hourly_frame`.

    Returns
    -------
    Panel
        Panel con la `spec` de `hourly_spec`.
    """
    return Panel(df=make_hourly_frame(**kwargs), spec=hourly_spec())  # type: ignore[arg-type]


def autocorrelation(values: np.ndarray, lag: int) -> float:
    """Autocorrelacion muestral de una serie a un retardo dado.

    Parameters
    ----------
    values
        Serie unidimensional.
    lag
        Retardo en pasos, mayor que cero.

    Returns
    -------
    float
        Coeficiente de correlacion de Pearson entre la serie y su version
        retardada.
    """
    centred = values - values.mean()
    head, tail = centred[:-lag], centred[lag:]
    denominator = float(np.sqrt((head**2).sum() * (tail**2).sum()))
    return float((head * tail).sum() / denominator)
