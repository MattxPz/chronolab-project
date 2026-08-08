"""Figuras puras: reciben DataFrames y devuelven `Figure`, sin E/S ni estado.

Cada tema de la EDA se resuelve como un par de funciones: ``compute_*`` hace el
calculo numerico y devuelve una trama ordenada; ``plot_*`` toma esa trama ya
calculada y devuelve una `plotly.graph_objects.Figure`, sin volver a calcular
nada. Los ``plot_*`` son "figuras puras" en el sentido estricto de
docs/ARCHITECTURE.md; los ``compute_*`` son el vecino pragmatico que hace falta
para que este modulo sea, de verdad, "las funciones reutilizables" de la EDA en
un solo sitio, en vez de estar repartido entre aqui y un modulo de diagnostico
que la arquitectura no habia previsto. Es una ampliacion deliberada del
alcance declarado de este modulo; se documenta aqui y en el resumen de la
tarea.

Ningun ``plot_*`` escribe en disco. Guardar en ``reports/figures/`` es
responsabilidad de quien llama (la notebook), con ``fig.write_image(...)``.

## Paleta

Los colores son la instancia validada del skill de diseno de datos del
proyecto (ocho tonos categoricos con separacion CVD >= 8 en OKLab, mas una
rampa secuencial de un solo tono y un par divergente). No se recalculan aqui:
se copian tal cual. Si cambia la paleta de referencia, este es el unico sitio
que hay que tocar.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    "anomaly_threshold",
    "compute_acf_pacf",
    "compute_degree_days",
    "compute_difficulty_table",
    "compute_hour_dow_matrix",
    "compute_lowess_fit",
    "compute_monthly_profile",
    "compute_mstl",
    "compute_periodogram",
    "compute_resampled_mean",
    "compute_series_difficulty",
    "model_color_map",
    "plot_accuracy_vs_cost",
    "plot_acf_pacf",
    "plot_anomaly_series",
    "plot_degree_days_correlation",
    "plot_difficulty_table",
    "plot_dm_heatmap",
    "plot_dst_continuity",
    "plot_forecast_overlay",
    "plot_holiday_effect",
    "plot_hour_dow_heatmap",
    "plot_monthly_profile",
    "plot_mstl",
    "plot_periodogram",
    "plot_prediction_decomposition",
    "plot_quality_overview",
    "plot_residuals",
    "plot_series_with_flags",
    "plot_temperature_scatter",
    "plot_tft_temporal_attention",
    "plot_tft_variable_attention",
    "series_color_map",
]

# --------------------------------------------------------------------------- #
# Paleta validada (skill de dataviz, references/palette.md). Modo claro: es lo
# que produce `fig.write_image(...)` para reports/figures/.
# --------------------------------------------------------------------------- #

CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # 1 azul
    "#eb6834",  # 2 naranja
    "#1baf7a",  # 3 aguamarina
    "#eda100",  # 4 amarillo
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 verde
    "#4a3aa7",  # 7 violeta
    "#e34948",  # 8 rojo
)
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb",
    "#9ec5f4",
    "#5598e7",
    "#2a78d6",
    "#1c5cab",
    "#104281",
    "#0d366b",
)
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
DIVERGING_RED_BLUE: tuple[str, ...] = (
    "#8f2020",
    "#b03030",
    "#e34948",
    "#ef8b8a",
    "#f0efec",
    "#9ec5f4",
    "#5598e7",
    "#2a78d6",
    "#104281",
)
"""Rampa divergente de nueve pasos, centrada en el gris neutro del quinto.

Misma rampa que ya usan las figuras estaticas de `scripts/run_anomaly_eval.py`
(``09_anomaly_heatmap.png``): un rojo (perder) - azul (ganar) simetrico, para
estadisticos con signo con cero significativo (Diebold-Mariano) o con un punto
neutro conocido, nunca para magnitudes sin signo (eso es `SEQUENTIAL_BLUE`)."""
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
_FONT_FAMILY = "system-ui, -apple-system, Segoe UI, sans-serif"


def series_color_map(series_ids: Sequence[str]) -> dict[str, str]:
    """Asigna un color categorico fijo a cada identificador de serie, en orden.

    El color sigue a la entidad, nunca a su posicion tras un filtro: llamar
    dos veces con el mismo conjunto (en el mismo orden) siempre da el mismo
    mapa, y es responsabilidad de quien llama mantener ese orden estable a lo
    largo de una notebook.

    Parameters
    ----------
    series_ids
        Identificadores de serie, en el orden en que deben tomar color. Mas de
        ocho no esta soportado por la paleta categorica validada: series de
        mas alla del octavo puesto repiten el ultimo color con una nota en el
        titulo de quien llame, no una advertencia silenciosa aqui.

    Returns
    -------
    dict[str, str]
        Mapa ``{unique_id: color_hex}``.
    """
    return {
        series_id: CATEGORICAL[min(i, len(CATEGORICAL) - 1)]
        for i, series_id in enumerate(series_ids)
    }


def model_color_map(model_ids: Sequence[str]) -> dict[str, str]:
    """Asigna un color categorico fijo a cada modelo, en orden.

    Misma regla que `series_color_map`, sobre la misma paleta: un modelo tiene
    un unico color en todas las figuras que lo mencionan (Forecast,
    Leaderboard), y ese color no depende de que otros modelos esten
    seleccionados en el momento.

    Parameters
    ----------
    model_ids
        Identificadores de modelo, en el orden en que deben tomar color.

    Returns
    -------
    dict[str, str]
        Mapa ``{model_id: color_hex}``.
    """
    return {
        model_id: CATEGORICAL[min(i, len(CATEGORICAL) - 1)] for i, model_id in enumerate(model_ids)
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convierte un color hex a una cadena `rgba(...)` de Plotly con opacidad dada."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(
    *, title: str | None = None, xaxis_title: str | None = None, yaxis_title: str | None = None
) -> dict[str, object]:
    """Layout compartido: tipografia, superficie y rejilla de la paleta validada."""
    layout: dict[str, object] = {
        "template": "plotly_white",
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": _FONT_FAMILY, "color": INK_PRIMARY, "size": 13},
        "xaxis": {
            "title": xaxis_title,
            "gridcolor": GRIDLINE,
            "linecolor": BASELINE,
            "tickfont": {"color": INK_MUTED},
        },
        "yaxis": {
            "title": yaxis_title,
            "gridcolor": GRIDLINE,
            "linecolor": BASELINE,
            "tickfont": {"color": INK_MUTED},
        },
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 60, "r": 30, "t": 50 if title else 20, "b": 50},
    }
    if title:
        layout["title"] = {"text": title, "font": {"size": 15, "color": INK_PRIMARY}}
    return layout


# --------------------------------------------------------------------------- #
# 1. Perfil de calidad de datos
# --------------------------------------------------------------------------- #


def plot_quality_overview(report: pd.DataFrame) -> go.Figure:
    """Barras agrupadas con las incidencias de calidad por serie.

    Parameters
    ----------
    report
        Salida de `chronolab.data.quality.coverage_report`: una fila por
        serie con ``n_gaps``, ``n_duplicated_pairs``, ``n_zeros`` y
        ``n_outliers``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    issue_columns = (
        ("n_gaps", "Huecos"),
        ("n_duplicated_pairs", "Duplicados"),
        ("n_zeros", "Ceros"),
        ("n_outliers", "Atipicos (z robusto)"),
    )
    # Se evita el cuarto slot de la paleta (amarillo, junto a naranja) para no
    # sentar dos tonos que comparten el gate de adyacencia mas justo.
    colors = (CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[6])

    fig = go.Figure()
    for (column, label), color in zip(issue_columns, colors, strict=True):
        fig.add_bar(
            x=report["unique_id"],
            y=report[column],
            name=label,
            marker_color=color,
            marker_line_width=0,
        )
    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        **_base_layout(title="Incidencias de calidad por serie", yaxis_title="Filas afectadas"),
    )
    return fig


def plot_series_with_flags(
    series_frame: pd.DataFrame, outliers: pd.DataFrame, *, unique_id: str, color: str
) -> go.Figure:
    """Serie temporal con los atipicos marcados; los huecos quedan como cortes en la linea.

    Parameters
    ----------
    series_frame
        Trama alineada (rejilla completa) de **una sola serie**, columnas
        ``ds`` e ``y``. Los `NaN` de `y` se dibujan como corte de la linea
        (`connectgaps=False`), que es la forma honesta de mostrar un hueco: no
        rellenarlo visualmente.
    outliers
        Salida de `chronolab.data.quality.detect_outliers` (puede incluir
        otras series; se filtra por `unique_id`).
    unique_id
        Identificador de la serie, usado en el titulo.
    color
        Color de la linea, tipicamente de `series_color_map`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ordered = series_frame.sort_values("ds")
    fig = go.Figure()
    fig.add_scatter(
        x=ordered["ds"],
        y=ordered["y"],
        mode="lines",
        line={"width": 2, "color": color},
        name=unique_id,
        connectgaps=False,
        showlegend=False,
    )
    series_outliers = outliers.loc[outliers["unique_id"] == unique_id]
    if len(series_outliers):
        fig.add_scatter(
            x=series_outliers["ds"],
            y=series_outliers["y"],
            mode="markers",
            marker={"size": 9, "color": STATUS["critical"], "line": {"width": 2, "color": SURFACE}},
            name="Atipico",
        )
    fig.update_layout(
        **_base_layout(title=f"{unique_id}: serie con huecos y atipicos", yaxis_title="y")
    )
    return fig


_DEFAULT_DST_WINDOW = pd.Timedelta(hours=8)


def plot_dst_continuity(
    aligned_frame: pd.DataFrame,
    *,
    unique_id: str,
    transition: pd.Timestamp,
    window: pd.Timedelta = _DEFAULT_DST_WINDOW,
    color: str = CATEGORICAL[0],
) -> go.Figure:
    """Zoom horario alrededor de una transicion de DST, ya en UTC alineado.

    Si el pipeline maneja bien el cambio de hora, esta figura debe mostrar
    puntos equiespaciados una hora, sin huecos ni duplicados, atravesando la
    medianoche de la fecha de transicion sin ninguna marca especial: en UTC
    la transicion no deja rastro.

    Parameters
    ----------
    aligned_frame
        Trama alineada (UTC ingenuo, rejilla completa).
    unique_id
        Serie a mostrar.
    transition
        Fecha civil (medianoche UTC) de la transicion.
    window
        Margen a cada lado del dia de transicion.
    color
        Color de la linea.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    start = pd.Timestamp(transition) - window
    end = pd.Timestamp(transition) + pd.Timedelta(days=1) + window
    sub = aligned_frame[
        (aligned_frame["unique_id"] == unique_id)
        & (aligned_frame["ds"] >= start)
        & (aligned_frame["ds"] < end)
    ].sort_values("ds")

    fig = go.Figure()
    fig.add_scatter(
        x=sub["ds"],
        y=sub["y"],
        mode="lines+markers",
        line={"width": 2, "color": color},
        marker={"size": 8, "color": color, "line": {"width": 2, "color": SURFACE}},
        showlegend=False,
    )
    fig.update_layout(
        **_base_layout(
            title=f"{unique_id}: continuidad horaria en UTC alrededor de {pd.Timestamp(transition).date()}",
            yaxis_title="y",
        )
    )
    return fig


# --------------------------------------------------------------------------- #
# 2. Descomposicion MSTL
# --------------------------------------------------------------------------- #


def compute_mstl(series: pd.Series, *, periods: Sequence[int] = (24, 168)) -> pd.DataFrame:
    """Descompone una serie en tendencia, una estacionalidad por periodo y residuo.

    Parameters
    ----------
    series
        Serie indexada por tiempo, con rejilla regular. Los `NaN` (huecos) se
        interpolan linealmente antes de descomponer: MSTL no admite huecos, y
        la alternativa de descartarlos rompería la rejilla regular que
        necesita.
    periods
        Longitudes estacionales en pasos, de la mas corta a la mas larga.

    Returns
    -------
    pandas.DataFrame
        Indexado igual que `series`, con columnas ``observed``, ``trend``,
        ``seasonal_<periodo>`` por cada periodo y ``resid``.
    """
    from statsmodels.tsa.seasonal import MSTL

    clean = series.interpolate(limit_direction="both")
    result = MSTL(clean, periods=list(periods)).fit()

    components = pd.DataFrame({"observed": clean, "trend": result.trend}, index=clean.index)
    seasonal = result.seasonal
    if isinstance(seasonal, pd.Series):
        components[f"seasonal_{periods[0]}"] = seasonal
    else:
        for period in periods:
            components[f"seasonal_{period}"] = seasonal[f"seasonal_{period}"]
    components["resid"] = result.resid
    return components


def plot_mstl(components: pd.DataFrame, *, periods: Sequence[int] = (24, 168)) -> go.Figure:
    """Panel apilado: observado, tendencia, una estacionalidad por periodo y residuo.

    Parameters
    ----------
    components
        Salida de `compute_mstl`.
    periods
        Los mismos periodos usados en `compute_mstl`, en el mismo orden.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n_rows = 3 + len(periods)
    titles = (
        ["Observado", "Tendencia"]
        + [f"Estacionalidad ({period} pasos)" for period in periods]
        + ["Residuo"]
    )
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.035, subplot_titles=titles
    )

    fig.add_scatter(
        x=components.index,
        y=components["observed"],
        mode="lines",
        line={"width": 1.3, "color": INK_PRIMARY},
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=components.index,
        y=components["trend"],
        mode="lines",
        line={"width": 2, "color": CATEGORICAL[0]},
        row=2,
        col=1,
    )
    for i, period in enumerate(periods):
        fig.add_scatter(
            x=components.index,
            y=components[f"seasonal_{period}"],
            mode="lines",
            line={"width": 1.2, "color": CATEGORICAL[1 + i]},
            row=3 + i,
            col=1,
        )
    fig.add_scatter(
        x=components.index,
        y=components["resid"],
        mode="lines",
        line={"width": 1, "color": INK_MUTED},
        row=n_rows,
        col=1,
    )

    fig.update_layout(
        showlegend=False,
        height=165 * n_rows,
        **_base_layout(title="Descomposicion MSTL"),
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"size": 12, "color": INK_SECONDARY}
    return fig


# --------------------------------------------------------------------------- #
# 3. ACF, PACF y periodograma
# --------------------------------------------------------------------------- #


def compute_acf_pacf(series: pd.Series, *, max_lag: int = 200) -> pd.DataFrame:
    """Autocorrelacion y autocorrelacion parcial hasta `max_lag`.

    Parameters
    ----------
    series
        Serie indexada por tiempo. Los huecos (`NaN`) se descartan antes de
        calcular: a diferencia de MSTL, ACF/PACF no necesitan rejilla regular
        para calcularse (aunque la interpretacion de "retardo" asume que si
        la hay, que es el caso tras `reindex_to_full_grid`).
    max_lag
        Numero maximo de retardos.

    Returns
    -------
    pandas.DataFrame
        Columnas ``lag``, ``acf`` y ``pacf``.
    """
    from statsmodels.tsa.stattools import acf, pacf

    clean = series.dropna().to_numpy()
    acf_vals = acf(clean, nlags=max_lag, fft=True)
    pacf_vals = pacf(clean, nlags=max_lag, method="ywm")
    return pd.DataFrame({"lag": np.arange(max_lag + 1), "acf": acf_vals, "pacf": pacf_vals})


def plot_acf_pacf(table: pd.DataFrame, *, n_obs: int) -> go.Figure:
    """Paneles apilados de ACF y PACF con la banda de significancia al 95%.

    Parameters
    ----------
    table
        Salida de `compute_acf_pacf`.
    n_obs
        Numero de observaciones usadas para calcular la autocorrelacion:
        determina el ancho de la banda de significancia, ``+-1.96/sqrt(n)``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    confidence = 1.96 / np.sqrt(n_obs)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("ACF", "PACF")
    )

    fig.add_hrect(
        y0=-confidence, y1=confidence, fillcolor=GRIDLINE, opacity=0.7, line_width=0, row=1, col=1
    )
    fig.add_bar(
        x=table["lag"],
        y=table["acf"],
        marker_color=CATEGORICAL[0],
        marker_line_width=0,
        row=1,
        col=1,
    )
    fig.add_hrect(
        y0=-confidence, y1=confidence, fillcolor=GRIDLINE, opacity=0.7, line_width=0, row=2, col=1
    )
    fig.add_bar(
        x=table["lag"],
        y=table["pacf"],
        marker_color=CATEGORICAL[1],
        marker_line_width=0,
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Retardo (horas)", row=2, col=1)
    fig.update_layout(
        showlegend=False,
        height=520,
        **_base_layout(title="Autocorrelacion (ACF) y autocorrelacion parcial (PACF)"),
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"size": 12, "color": INK_SECONDARY}
    return fig


def compute_periodogram(series: pd.Series) -> pd.DataFrame:
    """Densidad espectral de potencia, reexpresada en periodo (horas) en vez de frecuencia.

    Parameters
    ----------
    series
        Serie indexada por tiempo, sin huecos (se descartan antes de calcular).

    Returns
    -------
    pandas.DataFrame
        Columnas ``frequency``, ``period`` (``1/frequency``) y ``power``,
        ordenada por frecuencia creciente, sin la componente de frecuencia
        cero (periodo infinito).
    """
    from scipy.signal import periodogram

    clean = series.dropna().to_numpy()
    freqs, power = periodogram(clean, fs=1.0, detrend="linear", scaling="spectrum")
    table = pd.DataFrame({"frequency": freqs, "power": power})
    table = table[table["frequency"] > 0].copy()
    table["period"] = 1.0 / table["frequency"]
    return table.reset_index(drop=True)


def plot_periodogram(
    table: pd.DataFrame,
    *,
    highlight_periods: Sequence[float] = (24.0, 168.0),
    max_period: float = 400.0,
) -> go.Figure:
    """Periodograma en escala log, con los periodos esperados marcados.

    Parameters
    ----------
    table
        Salida de `compute_periodogram`.
    highlight_periods
        Periodos (en las mismas unidades que `table["period"]`) a marcar con
        una linea vertical y una etiqueta.
    max_period
        Periodo maximo mostrado en el eje x: por debajo de un año no aporta
        legibilidad extenderlo mas.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    view = table[table["period"] <= max_period]
    fig = go.Figure()
    fig.add_scatter(
        x=view["period"], y=view["power"], mode="lines", line={"width": 2, "color": CATEGORICAL[0]}
    )
    for period in highlight_periods:
        fig.add_vline(
            x=period,
            line={"color": INK_MUTED, "width": 1, "dash": "dot"},
            annotation_text=f"{period:g}h",
            annotation_font={"color": INK_SECONDARY, "size": 11},
        )
    fig.update_layout(
        **_base_layout(
            title="Periodograma", xaxis_title="Periodo (horas)", yaxis_title="Densidad espectral"
        )
    )
    fig.update_yaxes(type="log")
    return fig


# --------------------------------------------------------------------------- #
# 4. Perfiles agregados
# --------------------------------------------------------------------------- #

_DOW_LABELS = ("Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom")
_MONTH_LABELS = ("Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic")


def compute_hour_dow_matrix(frame: pd.DataFrame, *, value_column: str = "y") -> pd.DataFrame:
    """Media de `value_column` por hora del dia y dia de la semana, **en hora local**.

    Parameters
    ----------
    frame
        Trama con columnas ``hour`` (0-23) y ``dayofweek`` (0=lunes) en hora
        local, tal como las produce
        `chronolab.data.calendar.calendar_features`. Deliberadamente no
        deriva estas columnas de `ds` aqui: `ds` esta en UTC en la trama
        alineada, y usar la hora UTC directamente desalinearia el perfil
        respecto al reloj de pared que de verdad sigue la demanda.
    value_column
        Columna a promediar.

    Returns
    -------
    pandas.DataFrame
        Filas 0-23 (hora), columnas lunes a domingo.
    """
    matrix = frame.pivot_table(
        index="hour", columns="dayofweek", values=value_column, aggfunc="mean"
    )
    matrix = matrix.reindex(index=range(24), columns=range(7))
    matrix.columns = pd.Index(_DOW_LABELS)
    return matrix


def plot_hour_dow_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap hora x dia de la semana, un solo tono secuencial (magnitud).

    Parameters
    ----------
    matrix
        Salida de `compute_hour_dow_matrix`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n_steps = len(SEQUENTIAL_BLUE)
    colorscale = [[i / (n_steps - 1), color] for i, color in enumerate(SEQUENTIAL_BLUE)]
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=colorscale,
            colorbar={
                "title": {"text": "y medio"},
                "outlinewidth": 0,
                "tickfont": {"color": INK_MUTED},
            },
            hovertemplate="%{x}, %{y}h — %{z:.1f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_layout(
        **_base_layout(
            title="Perfil medio por hora del dia y dia de la semana",
            xaxis_title="Dia de la semana",
            yaxis_title="Hora del dia",
        )
    )
    return fig


def plot_holiday_effect(frame: pd.DataFrame, *, value_column: str = "y") -> go.Figure:
    """Distribucion de la demanda en dias festivos frente a dias normales (forma de enfasis).

    Parameters
    ----------
    frame
        Trama con la columna booleana ``is_holiday`` (de `calendar_features`)
        y `value_column`.
    value_column
        Columna a comparar.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    for is_holiday, color, name, opacity in (
        (False, INK_MUTED, "Dia normal", 0.35),
        (True, CATEGORICAL[1], "Festivo", 0.85),
    ):
        values = frame.loc[frame["is_holiday"] == is_holiday, value_column]
        fig.add_box(
            y=values,
            name=name,
            marker_color=color,
            line_color=color,
            fillcolor=_hex_to_rgba(color, opacity),
            boxmean=True,
        )
    fig.update_layout(
        showlegend=False,
        **_base_layout(
            title="Demanda en dias festivos frente a dias normales", yaxis_title=value_column
        ),
    )
    return fig


def compute_monthly_profile(frame: pd.DataFrame, *, value_column: str = "y") -> pd.DataFrame:
    """Media y banda P10-P90 de `value_column` por mes, **en calendario local**.

    Parameters
    ----------
    frame
        Trama con columna ``month`` (1-12) en hora local, de `calendar_features`.
    value_column
        Columna a agregar.

    Returns
    -------
    pandas.DataFrame
        Columnas ``month``, ``mean``, ``p10``, ``p90``.
    """
    grouped = frame.groupby("month")[value_column]
    table = grouped.agg(mean="mean", p10=lambda s: s.quantile(0.1), p90=lambda s: s.quantile(0.9))
    return table.reindex(range(1, 13)).reset_index()


def plot_monthly_profile(table: pd.DataFrame) -> go.Figure:
    """Linea de media mensual con banda P10-P90.

    Parameters
    ----------
    table
        Salida de `compute_monthly_profile`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.add_scatter(
        x=table["month"],
        y=table["p90"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=table["month"],
        y=table["p10"],
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor=_hex_to_rgba(CATEGORICAL[0], 0.15),
        name="P10-P90",
    )
    fig.add_scatter(
        x=table["month"],
        y=table["mean"],
        mode="lines+markers",
        line={"width": 2, "color": CATEGORICAL[0]},
        marker={"size": 8, "color": CATEGORICAL[0], "line": {"width": 2, "color": SURFACE}},
        name="Media",
    )
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=list(_MONTH_LABELS))
    fig.update_layout(**_base_layout(title="Perfil mensual", xaxis_title="Mes", yaxis_title="y"))
    return fig


# --------------------------------------------------------------------------- #
# 5. Relacion demanda-temperatura
# --------------------------------------------------------------------------- #


def compute_lowess_fit(x: pd.Series, y: pd.Series, *, frac: float = 0.2) -> pd.DataFrame:
    """Ajuste LOWESS (no parametrico) de `y` en funcion de `x`.

    Se usa LOWESS y no un polinomio porque no impone de antemano la forma en
    U esperada: si la relacion no fuese en U, el ajuste no la fabricaria.

    Parameters
    ----------
    x, y
        Series de la misma longitud e indice. Las filas con `NaN` en
        cualquiera de las dos se descartan.
    frac
        Fraccion de los datos usada en cada ventana local de suavizado.

    Returns
    -------
    pandas.DataFrame
        Columnas ``x`` e ``y_fit``, ordenada por `x` creciente.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    valid = x.notna() & y.notna()
    fitted = lowess(y[valid], x[valid], frac=frac, return_sorted=True)
    return pd.DataFrame(fitted, columns=["x", "y_fit"])


def plot_temperature_scatter(
    frame: pd.DataFrame, fit: pd.DataFrame, *, color: str, title: str | None = None
) -> go.Figure:
    """Dispersión demanda-temperatura con el ajuste LOWESS superpuesto.

    Parameters
    ----------
    frame
        Trama con columnas ``temp_c`` e ``y``.
    fit
        Salida de `compute_lowess_fit` sobre las mismas columnas.
    color
        Color de los puntos, tipicamente de `series_color_map`.
    title
        Titulo de la figura. Si es `None`, se usa uno generico.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.add_scatter(
        x=frame["temp_c"],
        y=frame["y"],
        mode="markers",
        marker={"size": 4, "color": color, "opacity": 0.25},
        name="Observaciones",
        showlegend=False,
    )
    fig.add_scatter(
        x=fit["x"],
        y=fit["y_fit"],
        mode="lines",
        line={"width": 3, "color": INK_PRIMARY},
        name="Ajuste LOWESS",
    )
    fig.update_layout(
        **_base_layout(
            title=title or "Demanda frente a temperatura",
            xaxis_title="Temperatura (C)",
            yaxis_title="y",
        )
    )
    return fig


def compute_degree_days(
    weather_frame: pd.DataFrame, *, base_heating: float = 18.0, base_cooling: float = 22.0
) -> pd.DataFrame:
    """Grados-dia de calefaccion (HDD) y refrigeracion (CDD) a partir de temperatura horaria.

    Parameters
    ----------
    weather_frame
        Trama con columnas ``ds`` (horaria) y ``temp_c``, de una unica
        ubicacion.
    base_heating
        Temperatura base de calefaccion: por debajo de este umbral se asume
        que hace falta calentar.
    base_cooling
        Temperatura base de refrigeracion: por encima de este umbral se asume
        que hace falta enfriar. Debe ser mayor o igual que `base_heating`;
        entre ambas hay una zona de confort sin grados-dia.

    Returns
    -------
    pandas.DataFrame
        Una fila por dia civil (segun el indice de `ds`), con ``date``,
        ``temp_mean``, ``hdd`` y ``cdd``.
    """
    daily_temp = weather_frame.set_index("ds")["temp_c"].resample("D").mean()
    hdd = (base_heating - daily_temp).clip(lower=0.0)
    cdd = (daily_temp - base_cooling).clip(lower=0.0)
    return pd.DataFrame(
        {
            "date": daily_temp.index,
            "temp_mean": daily_temp.to_numpy(),
            "hdd": hdd.to_numpy(),
            "cdd": cdd.to_numpy(),
        }
    )


def compute_resampled_mean(
    frame: pd.DataFrame, *, freq: str = "D", value_column: str = "y"
) -> pd.DataFrame:
    """Media diaria (u otra frecuencia) de `value_column`, para cruzar con grados-dia.

    Parameters
    ----------
    frame
        Trama con columnas ``ds`` y `value_column`, de una unica serie.
    freq
        Alias de offset de pandas de la frecuencia destino.
    value_column
        Columna a promediar.

    Returns
    -------
    pandas.DataFrame
        Columnas ``date`` y ``<value_column>_mean``.
    """
    resampled = frame.set_index("ds")[value_column].resample(freq).mean()
    return pd.DataFrame({"date": resampled.index, f"{value_column}_mean": resampled.to_numpy()})


def plot_degree_days_correlation(
    degree_days: pd.DataFrame, daily_demand: pd.DataFrame, *, value_column: str = "y"
) -> go.Figure:
    """Correlacion de Pearson de HDD y CDD con la demanda diaria media.

    Parameters
    ----------
    degree_days
        Salida de `compute_degree_days`.
    daily_demand
        Salida de `compute_resampled_mean` para la misma serie y frecuencia
        diaria.
    value_column
        Columna de demanda usada para nombrar la columna de `daily_demand`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    demand_col = f"{value_column}_mean"
    merged = degree_days.merge(daily_demand, on="date", how="inner")
    corr_hdd = merged["hdd"].corr(merged[demand_col])
    corr_cdd = merged["cdd"].corr(merged[demand_col])

    fig = go.Figure()
    fig.add_bar(
        x=["Grados-dia de calefaccion (HDD)", "Grados-dia de refrigeracion (CDD)"],
        y=[corr_hdd, corr_cdd],
        marker_color=[CATEGORICAL[0], CATEGORICAL[1]],
        marker_line_width=0,
        width=0.45,
        text=[f"{corr_hdd:.2f}", f"{corr_cdd:.2f}"],
        textposition="outside",
        textfont={"color": INK_PRIMARY},
    )
    fig.update_yaxes(range=[-1.0, 1.0])
    fig.update_layout(
        **_base_layout(
            title="Correlacion de HDD/CDD con la demanda diaria media",
            yaxis_title="Correlacion de Pearson",
        )
    )
    return fig


# --------------------------------------------------------------------------- #
# 6. Estadisticos de dificultad de la serie
# --------------------------------------------------------------------------- #


def compute_series_difficulty(
    series: pd.Series, *, periods: Sequence[int] = (24, 168)
) -> dict[str, float]:
    """Fuerza de tendencia, fuerza estacional por periodo y entropia espectral.

    Formulas de Hyndman & Wang (2015) para fuerza de tendencia y estacional,
    generalizadas a MSTL con una fuerza por periodo (Bandara et al.): ambas
    comparan la varianza del residuo con la varianza de "residuo mas el
    componente de interes", y valen 0 si el componente no aporta nada, cerca
    de 1 si domina la serie. La entropia espectral es la entropia de Shannon
    de la densidad espectral normalizada a densidad de probabilidad, dividida
    por `log(n)` para acotarla a `[0, 1]`: cerca de 0 es un espectro
    concentrado en pocas frecuencias (muy predecible), cerca de 1 es un
    espectro plano (ruido blanco, muy dificil de predecir).

    Parameters
    ----------
    series
        Serie indexada por tiempo, con rejilla regular (los huecos se
        interpolan).
    periods
        Periodos estacionales a evaluar, los mismos que en `compute_mstl`.

    Returns
    -------
    dict[str, float]
        ``trend_strength``, ``seasonal_strength_<periodo>`` por cada periodo,
        y ``spectral_entropy``.
    """
    from scipy.signal import periodogram
    from statsmodels.tsa.seasonal import MSTL

    clean = series.interpolate(limit_direction="both").dropna()
    result = MSTL(clean, periods=list(periods)).fit()
    resid = result.resid
    trend = result.trend
    seasonal = result.seasonal

    var_resid = float(np.var(resid))
    stats: dict[str, float] = {
        "trend_strength": max(0.0, 1.0 - var_resid / float(np.var(trend + resid)))
    }
    for period in periods:
        component = seasonal if isinstance(seasonal, pd.Series) else seasonal[f"seasonal_{period}"]
        stats[f"seasonal_strength_{period}"] = max(
            0.0, 1.0 - var_resid / float(np.var(component + resid))
        )

    _, power = periodogram(clean.to_numpy(), detrend="linear")
    power = power[1:]  # descarta la componente de frecuencia cero
    total_power = power.sum()
    if total_power <= 0:
        stats["spectral_entropy"] = float("nan")
    else:
        density = power / total_power
        density = density[density > 0]
        stats["spectral_entropy"] = float(-np.sum(density * np.log(density)) / np.log(len(density)))

    return stats


def compute_difficulty_table(
    series_map: Mapping[str, pd.Series], *, periods: Sequence[int] = (24, 168)
) -> pd.DataFrame:
    """Tabla comparativa de estadisticos de dificultad para varias series.

    Parameters
    ----------
    series_map
        Mapa ``{unique_id: serie}``, cada serie indexada por tiempo con
        rejilla regular.
    periods
        Se reenvia a `compute_series_difficulty`.

    Returns
    -------
    pandas.DataFrame
        Una fila por serie, con las columnas de `compute_series_difficulty`
        mas ``unique_id``.
    """
    rows = [
        {"unique_id": unique_id, **compute_series_difficulty(series, periods=periods)}
        for unique_id, series in series_map.items()
    ]
    return pd.DataFrame(rows)


def plot_difficulty_table(table: pd.DataFrame) -> go.Figure:
    """Tabla visual de los estadisticos de dificultad, para guardar como figura.

    Parameters
    ----------
    table
        Salida de `compute_difficulty_table`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    value_columns = [c for c in table.columns if c != "unique_id"]
    header_labels = ["Serie"] + [c.replace("_", " ").capitalize() for c in value_columns]
    cell_values = [table["unique_id"]] + [table[c].round(3) for c in value_columns]

    fig = go.Figure(
        go.Table(
            header={
                "values": header_labels,
                "fill_color": SURFACE,
                "font": {"color": INK_PRIMARY, "size": 13},
                "align": "left",
                "line": {"color": GRIDLINE},
            },
            cells={
                "values": cell_values,
                "fill_color": SURFACE,
                "font": {"color": INK_SECONDARY, "size": 12},
                "align": "left",
                "line": {"color": GRIDLINE},
                "height": 28,
            },
        )
    )
    fig.update_layout(
        height=90 + 40 * len(table), **_base_layout(title="Estadisticos de dificultad de la serie")
    )
    return fig


# --------------------------------------------------------------------------- #
# 7. Forecast: prediccion superpuesta, residuos
# --------------------------------------------------------------------------- #


def plot_forecast_overlay(
    context: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    model_colors: Mapping[str, str],
    quantile_low: str = "q_0250",
    quantile_high: str = "q_9750",
    show_bands: bool = True,
    unique_id: str | None = None,
) -> go.Figure:
    """Historico mas prediccion de uno o varios modelos, con banda de incertidumbre.

    Parameters
    ----------
    context
        Tramo previo al cutoff de **una sola serie**, columnas ``ds`` e ``y``.
        Se dibuja en gris neutro: es lo que el modelo pudo ver, no lo que se
        evalua.
    forecasts
        Filas de `chronolab.artifacts.reader.load_forecasts`, ya filtradas a
        una serie y a los modelos a mostrar. Columnas ``model_id``, ``ds``,
        ``y_hat`` y, si estan, ``y`` (el valor real del tramo evaluado, igual
        para todos los modelos) y las columnas de cuantil.
    model_colors
        Mapa ``{model_id: color}``, de `model_color_map`. Un modelo conserva
        su color aunque se quiten o anadan otros del multiselector.
    quantile_low, quantile_high
        Columnas de `forecasts` que delimitan la banda sombreada. Si faltan
        para un modelo (no soporta cuantiles), ese modelo se dibuja sin banda,
        nunca con una inventada.
    show_bands
        Conmutador de la banda de incertidumbre.
    unique_id
        Identificador de la serie, solo para el titulo.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    ordered_context = context.sort_values("ds")
    fig.add_scatter(
        x=ordered_context["ds"],
        y=ordered_context["y"],
        mode="lines",
        line={"width": 1.5, "color": INK_MUTED},
        name="Historico",
        connectgaps=False,
    )

    if "y" in forecasts.columns and not forecasts.empty:
        actual = forecasts[["ds", "y"]].drop_duplicates(subset="ds").sort_values("ds")
        fig.add_scatter(
            x=actual["ds"],
            y=actual["y"],
            mode="lines",
            line={"width": 2.2, "color": INK_PRIMARY},
            name="Real (evaluado)",
        )

    for model_id, group in forecasts.groupby("model_id", sort=False):
        color = model_colors.get(str(model_id), INK_MUTED)
        ordered = group.sort_values("ds")
        has_band = quantile_low in ordered.columns and quantile_high in ordered.columns
        if show_bands and has_band and ordered[quantile_low].notna().any():
            fig.add_scatter(
                x=ordered["ds"],
                y=ordered[quantile_high],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
            fig.add_scatter(
                x=ordered["ds"],
                y=ordered[quantile_low],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_hex_to_rgba(color, 0.18),
                showlegend=False,
                hoverinfo="skip",
                name=f"{model_id}: banda",
            )
        fig.add_scatter(
            x=ordered["ds"],
            y=ordered["y_hat"],
            mode="lines+markers",
            line={"width": 2, "color": color, "dash": "dot"},
            marker={"size": 5, "color": color},
            name=str(model_id),
        )

    title = (
        f"{unique_id}: prediccion frente a historico"
        if unique_id
        else "Prediccion frente a historico"
    )
    fig.update_layout(**_base_layout(title=title, yaxis_title="y"))
    return fig


def plot_residuals(forecasts: pd.DataFrame, *, model_colors: Mapping[str, str]) -> go.Figure:
    """Residuo (real menos prediccion) por instante, un color por modelo.

    Parameters
    ----------
    forecasts
        Filas con ``model_id``, ``ds``, ``y`` y ``y_hat``. La resta ocurre
        aqui, sobre columnas ya persistidas por el backtest -no es una metrica
        nueva, es la misma que dibuja cualquier grafico de residuos.
    model_colors
        Mapa ``{model_id: color}``, de `model_color_map`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.add_hline(y=0, line={"color": BASELINE, "width": 1})
    for model_id, group in forecasts.groupby("model_id", sort=False):
        ordered = group.sort_values("ds")
        color = model_colors.get(str(model_id), INK_MUTED)
        fig.add_scatter(
            x=ordered["ds"],
            y=ordered["y"] - ordered["y_hat"],
            mode="markers",
            marker={"size": 6, "color": color, "opacity": 0.75},
            name=str(model_id),
        )
    fig.update_layout(**_base_layout(title="Residuos (real - prediccion)", yaxis_title="y - y_hat"))
    return fig


# --------------------------------------------------------------------------- #
# 8. Leaderboard: precision vs coste, Diebold-Mariano
# --------------------------------------------------------------------------- #


def plot_accuracy_vs_cost(
    leaderboard: pd.DataFrame,
    *,
    model_colors: Mapping[str, str],
    x_column: str = "fit_seconds_total",
    y_column: str = "mase",
) -> go.Figure:
    """Dispersión precision (eje y, mas bajo es mejor) frente a coste (eje x, log).

    Parameters
    ----------
    leaderboard
        Una fila por modelo (ya filtrada a una etapa y a la agregacion sobre
        todas las series, ``unique_id`` nulo). Columnas ``model_id``,
        `x_column` e `y_column`.
    model_colors
        Mapa ``{model_id: color}``, de `model_color_map`.
    x_column, y_column
        Columnas de coste y de precision a graficar.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    safe_x = leaderboard[x_column].clip(lower=1e-3)
    for (_, row), x in zip(leaderboard.iterrows(), safe_x, strict=True):
        model_id = str(row["model_id"])
        color = model_colors.get(model_id, INK_MUTED)
        fig.add_scatter(
            x=[x],
            y=[row[y_column]],
            mode="markers+text",
            marker={"size": 14, "color": color, "line": {"width": 2, "color": SURFACE}},
            text=[model_id],
            textposition="top center",
            textfont={"size": 10, "color": INK_SECONDARY},
            name=model_id,
            showlegend=False,
            hovertemplate=f"{model_id}<br>{x_column}=%{{x:.3g}}s<br>{y_column}=%{{y:.4f}}<extra></extra>",
        )
    fig.update_xaxes(type="log", title_text=f"{x_column} (s, escala log)")
    fig.update_layout(
        **_base_layout(title="Precision frente a coste computacional", yaxis_title=y_column.upper())
    )
    return fig


def plot_dm_heatmap(dm_matrix: pd.DataFrame) -> go.Figure:
    """Matriz de estadisticos de Diebold-Mariano, un color por signo y magnitud.

    Parameters
    ----------
    dm_matrix
        Salida de `chronolab.artifacts.reader.load_dm_matrix`: una fila por
        pareja ordenada ``(model_a, model_b)`` con ``stat`` y ``p_value``.
        Negativo en la celda ``(fila, columna)`` significa que el modelo de la
        fila pierde menos que el de la columna (`chronolab.evaluation.stats_tests.DMResult`).

    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap con la rampa `DIVERGING_RED_BLUE`, centrado en cero, y el
        estadistico mas el p-valor anotados en cada celda.
    """
    models = sorted(set(dm_matrix["model_a"].astype(str)) | set(dm_matrix["model_b"].astype(str)))
    slot = {model: i for i, model in enumerate(models)}
    stat = np.zeros((len(models), len(models)))
    text = np.full((len(models), len(models)), "-", dtype=object)

    for row in dm_matrix.itertuples(index=False):
        i, j = slot[str(row.model_a)], slot[str(row.model_b)]
        stat_value = float(row.stat)  # type: ignore[arg-type]
        stat[i, j] = stat_value
        p_value = row.p_value
        text[i, j] = (
            f"{stat_value:.2f}<br>p={float(p_value):.3f}"  # type: ignore[arg-type]
            if pd.notna(p_value)
            else "n/d"
        )

    bound = float(np.nanmax(np.abs(stat))) if models else 1.0
    bound = bound if bound > 0 else 1.0
    n_steps = len(DIVERGING_RED_BLUE)
    colorscale = [[i / (n_steps - 1), color] for i, color in enumerate(DIVERGING_RED_BLUE)]

    fig = go.Figure(
        go.Heatmap(
            z=stat,
            x=models,
            y=models,
            colorscale=colorscale,
            zmid=0,
            zmin=-bound,
            zmax=bound,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10, "color": INK_PRIMARY},
            colorbar={
                "title": {"text": "Estadistico DM"},
                "outlinewidth": 0,
                "tickfont": {"color": INK_MUTED},
            },
            hovertemplate="fila %{y} vs columna %{x}: stat=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(**_base_layout(title="Diebold-Mariano: significancia por pareja de modelos"))
    return fig


# --------------------------------------------------------------------------- #
# 9. Anomalias: serie marcada, umbral sobre scores ya calculados
# --------------------------------------------------------------------------- #


def anomaly_threshold(alpha: float) -> float:
    """Umbral de score correspondiente a un nivel alfa.

    Los cuatro detectores del hito de anomalias emiten ``-log10(p)`` como
    score (``scripts/run_anomaly_eval.py``), asi que el mismo umbral
    ``-log10(alpha)`` es comparable entre ellos: no es una eleccion de este
    modulo, es una propiedad de diseno de los detectores que aqui solo se
    expresa. Recalcularlo sobre scores ya calculados -sin volver a puntuar
    nada- es exactamente lo que permite mover el slider de alfa (A5).

    Parameters
    ----------
    alpha
        Nivel de significancia, en ``(0, 1)``.

    Returns
    -------
    float
        Umbral: un score ``>=`` este valor se marca como anomalia.

    Raises
    ------
    ValueError
        Si `alpha` no esta en ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha debe estar en (0, 1): {alpha}")
    return float(-np.log10(alpha))


def plot_anomaly_series(
    series: pd.DataFrame,
    scores: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    threshold: float,
    color: str,
    unique_id: str,
) -> go.Figure:
    """Serie con los eventos reales sombreados y los puntos marcados por el detector.

    Parameters
    ----------
    series
        Trama de **una sola serie**, columnas ``ds`` e ``y``.
    scores
        Scores de **un solo detector** sobre esa serie: columnas ``ds``,
        ``score`` y ``scorable``.
    truth
        Subconjunto de `chronolab.artifacts.reader.load_anomaly_truth` para
        esa serie. Puede estar vacio (serie sin anomalias inyectadas).
    threshold
        Umbral vigente, de `anomaly_threshold`. Un punto se marca si
        ``scorable`` y ``score >= threshold``.
    color
        Color de la linea de la serie, tipicamente de `series_color_map`.
    unique_id
        Identificador de la serie, para el titulo.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ordered = series.sort_values("ds")
    fig = go.Figure()

    for _, group in truth.groupby("event_id"):
        fig.add_vrect(
            x0=group["ds"].min(),
            x1=group["ds"].max() + pd.Timedelta(hours=1),
            fillcolor=_hex_to_rgba(STATUS["warning"], 0.16),
            line_width=0,
            layer="below",
        )

    fig.add_scatter(
        x=ordered["ds"],
        y=ordered["y"],
        mode="lines",
        line={"width": 1.5, "color": color},
        name=unique_id,
        connectgaps=False,
    )

    merged = scores.merge(ordered[["ds", "y"]], on="ds", how="inner")
    flagged = merged.loc[merged["scorable"].fillna(False) & (merged["score"] >= threshold)]
    if len(flagged):
        fig.add_scatter(
            x=flagged["ds"],
            y=flagged["y"],
            mode="markers",
            marker={
                "size": 9,
                "color": STATUS["critical"],
                "symbol": "x",
                "line": {"width": 1.5, "color": SURFACE},
            },
            name=f"Marcado (score >= {threshold:.2f})",
        )

    fig.update_layout(
        **_base_layout(title=f"{unique_id}: serie con anomalias marcadas", yaxis_title="y")
    )
    return fig


# --------------------------------------------------------------------------- #
# 10. Explicabilidad: atencion del TFT, descomposicion de la prediccion
# --------------------------------------------------------------------------- #


def plot_tft_variable_attention(table: pd.DataFrame) -> go.Figure:
    """Peso de atencion por variable de entrada, agrupado por bloque pasado/futuro.

    Parameters
    ----------
    table
        Salida de `chronolab.artifacts.reader.load_tft_interpretability`,
        restringida internamente a ``kind == "attention_variable"``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    variable = table.loc[table["kind"] == "attention_variable"].dropna(subset=["feature", "block"])
    blocks = sorted(variable["block"].unique())
    block_labels = {"past": "Pasado", "future": "Futuro"}
    colors = {block: CATEGORICAL[i] for i, block in enumerate(blocks)}

    fig = go.Figure()
    for block in blocks:
        sub = variable.loc[variable["block"] == block].sort_values("value")
        fig.add_bar(
            x=sub["value"],
            y=sub["feature"],
            orientation="h",
            name=block_labels.get(block, block),
            marker_color=colors[block],
            marker_line_width=0,
        )
    fig.update_layout(
        barmode="group",
        **_base_layout(title="Importancia de variables (atencion del TFT)", xaxis_title="Peso"),
    )
    return fig


def plot_tft_temporal_attention(table: pd.DataFrame) -> go.Figure:
    """Peso de atencion temporal a lo largo de la ventana de contexto y horizonte.

    Parameters
    ----------
    table
        Salida de `chronolab.artifacts.reader.load_tft_interpretability`,
        restringida internamente a ``kind == "attention_temporal"``.

    Returns
    -------
    plotly.graph_objects.Figure
        El paso ``0`` (linea vertical punteada) es el cutoff: a la izquierda,
        contexto observado; a la derecha, horizonte de prediccion.
    """
    temporal = table.loc[table["kind"] == "attention_temporal"].dropna(subset=["offset"])
    temporal = temporal.sort_values("offset")

    fig = go.Figure()
    fig.add_scatter(
        x=temporal["offset"],
        y=temporal["value"],
        mode="lines",
        line={"width": 2, "color": CATEGORICAL[0]},
        fill="tozeroy",
        fillcolor=_hex_to_rgba(CATEGORICAL[0], 0.18),
        showlegend=False,
    )
    fig.add_vline(
        x=0,
        line={"color": INK_MUTED, "width": 1, "dash": "dot"},
        annotation_text="cutoff",
        annotation_font={"color": INK_SECONDARY, "size": 11},
    )
    fig.update_layout(
        **_base_layout(
            title="Atencion temporal (TFT)",
            xaxis_title="Paso relativo al cutoff",
            yaxis_title="Peso de atencion",
        )
    )
    return fig


def plot_prediction_decomposition(
    components: pd.DataFrame, *, unique_id: str, ds: pd.Timestamp
) -> go.Figure:
    """Descompone el valor observado de un instante en tendencia, estacionales y residuo.

    Parameters
    ----------
    components
        Salida de `chronolab.artifacts.reader.load_mstl_components`,
        restringida a una sola serie.
    unique_id
        Identificador de la serie, para el titulo.
    ds
        Instante a descomponer. Debe existir en ``components["ds"]``.

    Returns
    -------
    plotly.graph_objects.Figure
        Cascada: cada barra es la contribucion de un componente, la ultima es
        el total observado.

    Raises
    ------
    KeyError
        Si `ds` no esta en `components`.
    """
    row = components.loc[components["ds"] == pd.Timestamp(ds)]
    if row.empty:
        raise KeyError(f"'{ds}' no esta en la descomposicion de '{unique_id}'")
    values = row.iloc[0]

    fig = go.Figure(
        go.Waterfall(
            x=["Tendencia", "Estacional (24h)", "Estacional (168h)", "Residuo", "Observado"],
            measure=["relative", "relative", "relative", "relative", "total"],
            y=[
                values["trend"],
                values["seasonal_24"],
                values["seasonal_168"],
                values["resid"],
                0,
            ],
            increasing={"marker": {"color": CATEGORICAL[2]}},
            decreasing={"marker": {"color": CATEGORICAL[7]}},
            totals={"marker": {"color": INK_PRIMARY}},
            connector={"line": {"color": BASELINE}},
        )
    )
    fig.update_layout(
        **_base_layout(
            title=f"{unique_id}: descomposicion de la prediccion en {pd.Timestamp(ds)}",
            yaxis_title="y",
        )
    )
    return fig
