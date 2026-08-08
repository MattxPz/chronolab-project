"""Evaluacion completa de deteccion: 4 detectores x 6 tipos de anomalia x 3 series.

Corre el hito de anomalias de punta a punta y deja tres artefactos:

1. `reports/results/anomaly_results.parquet` — tabla larga de metricas, en el
   formato de `chronolab.evaluation.anomaly_metrics.METRIC_COLUMNS`.
2. `reports/results/anomaly_scores.parquet` — los scores crudos de los cuatro
   detectores sobre el holdout, para poder redibujar sin repetir el backtest.
3. `reports/figures/09_anomaly_heatmap.png` y `09_anomaly_pr_curves.png`.

Uso: ``uv run --extra ml --extra deep python scripts/run_anomaly_eval.py``

Cuatro decisiones del diseno experimental, y por que
-----------------------------------------------------
**El modelo base es MSTL y no el naive estacional.** Dos de los cuatro
detectores puntuan residuos, asi que heredan la memoria del modelo base. El
naive estacional repite el valor de hace un ciclo: cada anomalia reaparece
como prediccion `season` pasos despues y genera un falso positivo **garantizado**
con el signo cambiado. Eso no mediria detectores, mediria el modelo base. MSTL
estima la componente estacional suavizando entre decenas de ciclos, asi que un
punto anomalo entra en ella diluido. Cuesta 4.2 s por ventana frente a 0.012;
se paga.

**Las anomalias se inyectan solo en el holdout.** Los cuatro detectores calibran
sobre un tramo limpio. Es una idealizacion —en produccion se calibra sobre datos
que ya contienen anomalias— pero afecta a los cuatro por igual y aisla lo que se
quiere medir. Los ultimos tramos del holdout si entrenan sobre anomalias
anteriores, porque el entrenamiento es deslizante.

**El plan es teselado (`step_size == h`) y reajusta cada ventana.**
`artifacts.reader.scoring_frame` lo exige: con solape, un instante tendria varias
predicciones y podria caer en dev y en holdout a la vez. `refit_every=1` lo exige
MSTL con intervalos conformales, cuyo horizonte queda fijado en el ajuste.

**El umbral es el mismo para los cuatro.** Los cuatro emiten `-log10(p)`, asi que
``score >= -log10(alpha)`` significa lo mismo en todos y la comparacion a alfa
fijo es legitima. Esa comparabilidad es una propiedad del diseno de los
detectores, no de este script.
"""

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tests.fixtures.synthetic import make_hourly_panel  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

from chronolab.anomaly.autoencoder import AutoencoderDetector  # noqa: E402
from chronolab.anomaly.conformal import ConformalDetector  # noqa: E402
from chronolab.anomaly.injection import KINDS, AnomalySpec, inject_anomalies  # noqa: E402
from chronolab.anomaly.isolation import IsolationForestDetector  # noqa: E402
from chronolab.anomaly.matrix_profile import MatrixProfileDetector  # noqa: E402
from chronolab.artifacts.reader import scoring_frame  # noqa: E402
from chronolab.evaluation.anomaly_metrics import (  # noqa: E402
    common_scorable_mask,
    evaluate_detector,
)
from chronolab.evaluation.backtest import BacktestPlan, backtest  # noqa: E402
from chronolab.models.adapters.statsforecast import MSTLForecaster  # noqa: E402
from chronolab.panel import Panel  # noqa: E402
from chronolab.types import DetectorId  # noqa: E402

RESULTS_DIR = ROOT / "reports" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_PATH = RESULTS_DIR / "anomaly_results.parquet"
SCORES_PATH = RESULTS_DIR / "anomaly_scores.parquet"
TRUTH_PATH = RESULTS_DIR / "anomaly_truth.parquet"

H = 24
"""Horizonte y, por el teselado, tambien el paso entre cutoffs."""

TRAIN_SIZE = 1344
"""Ocho semanas de entrenamiento deslizante: 8 ciclos del periodo mas largo (168)."""

DEV_WINDOWS = 45
"""Ventanas de calibracion: 1080 pasos por serie, limpios de anomalias."""

HOLDOUT_WINDOWS = 45
"""Ventanas puntuadas: 1080 pasos por serie, con las anomalias inyectadas."""

N_SERIES = 3
SEED = 20_240_807
ALPHA = 0.05
"""Nivel al que se binariza el score: se marca donde ``score >= -log10(alpha)``."""

MERGE_GAP = 2
"""Tolerancia de fusion, la misma que `anomaly.events` usa por defecto."""

VUS_MAX_BUFFER = 24
"""Tolerancia maxima de VUS-PR: un ciclo diario, el orden de duracion de un evento."""

EVENT_SPACING = 56
"""Pasos entre inicios de eventos consecutivos dentro de una serie."""

ROUNDS = 3
"""Repeticiones de los seis tipos por serie: 18 eventos por serie, 54 en total."""

# duracion base y magnitud por tipo; la duracion crece un paso por ronda para
# que la tabla no dependa de una unica longitud por tipo.
ANOMALY_PARAMS: dict[str, tuple[int, float]] = {
    "spike": (1, 7.0),
    "level_shift": (12, 3.5),
    "variance_shift": (18, 5.0),
    "seasonal_phase": (12, 12.0),
    "sensor_freeze": (10, 0.02),
    "data_gap": (5, 1.0),
}

DISPLAY_NAMES: dict[str, str] = {
    "conformal": "Conformal (CQR)",
    "isoforest": "IsolationForest",
    "lstm_ae": "LSTM-Autoencoder",
    "matrix_profile": "MatrixProfile",
}
"""Prefijo de `detector_id` -> nombre corto para figuras y tablas."""


def short_name(detector_id: str) -> str:
    """Nombre corto de un detector a partir de su identificador.

    Parameters
    ----------
    detector_id
        Identificador completo, que codifica los hiperparametros.

    Returns
    -------
    str
        Nombre legible, o el identificador entero si no se reconoce el prefijo.
    """
    for prefix, name in DISPLAY_NAMES.items():
        if detector_id.startswith(prefix):
            return name
    return detector_id


def build_plan() -> BacktestPlan:
    """Plan teselado con reajuste por ventana.

    Returns
    -------
    BacktestPlan
        `step_size == h` porque lo exige la deteccion, y ``refit_every=1``
        porque lo exigen los intervalos conformales de MSTL.
    """
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


def anomaly_specs(panel: Panel, holdout_start: pd.Timestamp) -> list[AnomalySpec]:
    """Calendario de inyeccion: los seis tipos, repetidos, sobre cada serie.

    Los tipos se intercalan en lugar de agruparse para que ninguno caiga
    sistematicamente al principio o al final del holdout, donde el detector
    tiene mas o menos contexto acumulado.

    Parameters
    ----------
    panel
        Panel limpio.
    holdout_start
        Primer instante puntuado.

    Returns
    -------
    list of AnomalySpec
        ``ROUNDS * 6`` eventos por serie.
    """
    specs: list[AnomalySpec] = []
    for series_index, uid in enumerate(panel.ids()):
        # Desfase por serie: sin el, las tres series tendrian la anomalia del
        # mismo tipo exactamente en el mismo instante y las series dejarian de
        # ser replicas independientes.
        offset = 30 + series_index * 11
        for index in range(ROUNDS * len(KINDS)):
            kind = KINDS[index % len(KINDS)]
            base_duration, magnitude = ANOMALY_PARAMS[kind]
            start = holdout_start + pd.Timedelta(hours=offset + index * EVENT_SPACING)
            specs.append(
                AnomalySpec(
                    kind=kind,
                    unique_id=str(uid),
                    start=start,
                    duration=base_duration + index // len(KINDS),
                    magnitude=magnitude,
                    direction="up" if (index // len(KINDS)) % 2 == 0 else "down",
                )
            )
    return specs


def build_contaminated_panel() -> tuple[Panel, pd.DataFrame, BacktestPlan]:
    """Panel sintetico con las anomalias inyectadas y su verdad.

    Las ventanas se calculan **antes** de inyectar: el splitter solo mira la
    rejilla temporal, nunca los valores, asi que las mismas ventanas salen del
    panel limpio y del contaminado. Eso permite saber donde empieza el holdout
    sin haber contaminado nada todavia.

    Returns
    -------
    tuple
        Panel contaminado, tabla de verdad y plan.
    """
    plan = build_plan()
    clean = make_hourly_panel(
        n_series=N_SERIES,
        n_hours=TRAIN_SIZE + (DEV_WINDOWS + HOLDOUT_WINDOWS) * H,
        seed=SEED,
    )
    windows = plan.splitter().split(clean)
    holdout = [window for window in windows if window.stage == "holdout"]
    print(
        f"ventanas: {len(windows)} ({len(windows) - len(holdout)} dev, {len(holdout)} holdout); "
        f"holdout {holdout[0].first_pred} .. {holdout[-1].last_pred}"
    )

    specs = anomaly_specs(clean, holdout[0].first_pred)
    contaminated, truth = inject_anomalies(clean, specs, seed=SEED)
    print(f"inyectados {len(specs)} eventos, {len(truth)} instantes anomalos")
    print(truth.groupby("anomaly_type").size().to_string())
    return contaminated, truth, plan


def run_backtest(panel: Panel, plan: BacktestPlan) -> object:
    """Backtest de MSTL sobre el panel contaminado.

    Parameters
    ----------
    panel
        Panel con anomalias.
    plan
        Plan teselado.

    Returns
    -------
    BacktestResult
        Artefactos en memoria del run.
    """
    model = MSTLForecaster(h=H, season_lengths=(24, 168))
    started = time.perf_counter()
    with warnings.catch_warnings():
        # statsforecast avisa de convergencia del AutoARIMA de tendencia en
        # algunas ventanas; el ajuste que devuelve sigue siendo utilizable y el
        # motor registraria un fallo real con status="failed".
        warnings.simplefilter("ignore", UserWarning)
        result = backtest(panel, [model], plan)
    elapsed = time.perf_counter() - started

    runs = result.model_runs
    failed = int((runs["status"] != "ok").sum())
    print(f"backtest MSTL: {elapsed:.1f}s, {len(runs)} ventanas, {failed} no-ok")
    if failed:
        print(runs.loc[runs["status"] != "ok", ["window_id", "status", "error"]].head().to_string())
    return result


def score_detectors(result: object) -> dict[DetectorId, pd.DataFrame]:
    """Calibra los cuatro detectores sobre dev y los puntua sobre holdout.

    Parameters
    ----------
    result
        Resultado del backtest.

    Returns
    -------
    dict
        ``detector_id -> scores``, todos sobre la misma rejilla del holdout.
    """
    model_id = MSTLForecaster(h=H).model_id
    calib = scoring_frame(result, model_id=model_id, stage="dev")  # type: ignore[arg-type]
    holdout = scoring_frame(result, model_id=model_id, stage="holdout")  # type: ignore[arg-type]
    print(f"calibracion {calib.start} .. {calib.end} ({len(calib.df)} filas)")
    print(f"puntuacion  {holdout.start} .. {holdout.end} ({len(holdout.df)} filas)")

    detectors = [
        ConformalDetector(base_model_id=model_id, alpha_nominal=ALPHA, alpha_ref=ALPHA),
        IsolationForestDetector(base_model_id=model_id, window=H, n_estimators=200, seed=SEED),
        AutoencoderDetector(
            seq_len=H, hidden_size=32, latent_size=8, epochs=40, seed=SEED, alpha_ref=ALPHA
        ),
        MatrixProfileDetector(m=H),
    ]

    scores: dict[DetectorId, pd.DataFrame] = {}
    for detector in detectors:
        started = time.perf_counter()
        fitted = detector.fit(calib)
        frame = fitted.score(holdout)
        elapsed = time.perf_counter() - started
        rate = float((frame["scorable"] & (frame["score"] >= -np.log10(ALPHA))).mean())
        print(
            f"  {short_name(str(detector.detector_id)):<18} {elapsed:6.1f}s  "
            f"scorable={frame['scorable'].mean():.3f}  marcado={rate:.4f}  "
            f"id={detector.detector_id}"
        )
        scores[detector.detector_id] = frame
    return scores


def evaluate(
    scores: dict[DetectorId, pd.DataFrame], truth: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalua los cuatro detectores sobre la mascara `scorable` comun.

    Parameters
    ----------
    scores
        Scores por detector.
    truth
        Tabla de verdad de la inyeccion.

    Returns
    -------
    tuple
        Tabla larga de metricas y la mascara comun usada.
    """
    support = common_scorable_mask(scores)
    own = {name: float(frame["scorable"].mean()) for name, frame in scores.items()}
    print(
        f"mascara comun: {support['scorable'].mean():.4f} de {len(support)} instantes "
        f"(propias: {', '.join(f'{short_name(str(k))}={v:.3f}' for k, v in own.items())})"
    )

    tables = [
        evaluate_detector(
            frame,
            truth,
            detector_id=detector_id,
            support=support,
            alpha=ALPHA,
            merge_gap=MERGE_GAP,
            vus_max_buffer=VUS_MAX_BUFFER,
        )
        for detector_id, frame in scores.items()
    ]
    return pd.concat(tables, ignore_index=True), support


def summarise(table: pd.DataFrame) -> None:
    """Imprime el resumen que se lee para escribir los hallazgos.

    Parameters
    ----------
    table
        Tabla larga de metricas.
    """
    pooled = table.loc[table["unique_id"].isna()]

    print("\n=== Global (anomaly_type = all) ===")
    overall = pooled.loc[pooled["anomaly_type"] == "all"].pivot_table(
        index="detector_id", columns="metric", values="value"
    )
    columns = [
        "range_precision",
        "range_recall",
        "range_f1",
        "affiliation_precision",
        "affiliation_recall",
        "affiliation_f1",
        "auc_pr",
        "auc_pr_baseline",
        "vus_pr",
        "point_f1",
        "detection_rate",
        "detection_delay_mean",
        "false_alarms_per_1000",
        "n_false_alarm_events",
        "n_support_gaps",
    ]
    overall.index = [short_name(str(name)) for name in overall.index]
    print(overall[[c for c in columns if c in overall.columns]].round(4).to_string())

    for metric in ("range_recall", "affiliation_recall", "detection_rate", "auc_pr"):
        print(f"\n=== {metric} por tipo ===")
        grid = pooled.loc[
            (pooled["metric"] == metric) & (pooled["anomaly_type"] != "all")
        ].pivot_table(index="detector_id", columns="anomaly_type", values="value")
        grid.index = [short_name(str(name)) for name in grid.index]
        print(grid.round(4).to_string())

    print("\n=== eventos por tipo (n_events, agregado) ===")
    events = pooled.loc[
        (pooled["metric"] == "range_recall") & (pooled["anomaly_type"] != "all")
    ].pivot_table(index="detector_id", columns="anomaly_type", values="n_events")
    print(events.head(1).round(0).to_string())


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PLANE = "#f9f9f7"

SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
"""Ranuras categoricas 1-4. Se asignan por **identidad** de detector, en el orden
de `DISPLAY_NAMES`, nunca por su puesto en el ranking: si en otra ejecucion cambia
quien gana, cada detector conserva su color."""

SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
"""Rampa secuencial de un solo tono, claro -> oscuro. Nunca un arcoiris."""

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
"""Polos calido/frio con gris neutro en el centro, mismo numero de pasos por brazo."""

ORDERED_TYPES: tuple[str, ...] = (
    "spike",
    "level_shift",
    "variance_shift",
    "seasonal_phase",
    "sensor_freeze",
    "data_gap",
)
TYPE_LABELS: dict[str, str] = {
    "spike": "pico",
    "level_shift": "escalon",
    "variance_shift": "varianza",
    "seasonal_phase": "desfase",
    "sensor_freeze": "congelado",
    "data_gap": "hueco",
}


def _grid(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rejilla detector x tipo de una metrica, agregada sobre series.

    Parameters
    ----------
    table
        Tabla larga de metricas.
    metric
        Metrica a extraer.

    Returns
    -------
    pandas.DataFrame
        Filas en el orden de `DISPLAY_NAMES`, columnas en `ORDERED_TYPES`.
    """
    pooled = table.loc[
        table["unique_id"].isna() & (table["metric"] == metric) & (table["anomaly_type"] != "all")
    ]
    grid = pooled.pivot_table(
        index="detector_id", columns="anomaly_type", values="value", dropna=False
    )
    grid.index = [short_name(str(name)) for name in grid.index]
    order = [name for name in DISPLAY_NAMES.values() if name in grid.index]
    # Reindexado sobre los seis tipos, no interseccion: `pivot_table` elimina una
    # columna entera si todos sus valores son `NaN`, y un tipo que **ningun**
    # detector captura desapareceria del mapa. Esa ausencia es justo el resultado
    # que hay que poder ver, asi que la columna se conserva y sus celdas dicen
    # "n/d".
    return grid.loc[order].reindex(columns=list(ORDERED_TYPES))


def _heatmap_panel(
    axes: plt.Axes,
    grid: pd.DataFrame,
    *,
    colours: Sequence[str],
    vmin: float,
    vmax: float,
    title: str,
    subtitle: str,
) -> ScalarMappable:
    """Dibuja un panel del mapa de calor con las celdas etiquetadas.

    Etiquetar **todas** las celdas no contradice la regla de no poner un numero
    en cada punto: en un mapa de calor la etiqueta *es* la vista de tabla, y sin
    ella el valor quedaria codificado solo en color, que es lo que la regla de
    accesibilidad prohibe.

    Parameters
    ----------
    axes
        Ejes de destino.
    grid
        Rejilla detector x tipo.
    colours
        Rampa de color.
    vmin, vmax
        Extremos de la escala.
    title, subtitle
        Titulo y aclaracion de lectura del panel.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        Para construir la barra de color.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("chronolab", list(colours))
    norm = Normalize(vmin=vmin, vmax=vmax)
    values = grid.to_numpy(dtype=float)

    axes.set_xlim(0, grid.shape[1])
    axes.set_ylim(grid.shape[0], 0)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = values[row, column]
            missing = not np.isfinite(value)
            # Hueco de 2 px hacia la superficie entre celdas, en vez de un borde
            # dibujado alrededor de cada una.
            axes.add_patch(
                Rectangle(
                    (column + 0.012, row + 0.02),
                    0.976,
                    0.96,
                    facecolor=PLANE if missing else cmap(norm(value)),
                    # El rayado hereda el color del borde; con el negro por
                    # defecto tapa la etiqueta que tiene que explicar la celda.
                    edgecolor=AXIS if missing else "none",
                    linewidth=0.0,
                    hatch="//" if missing else None,
                )
            )
            if missing:
                axes.text(
                    column + 0.5,
                    row + 0.5,
                    "n/d",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=INK_SECONDARY,
                    style="italic",
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.0},
                )
                continue
            # Tinta clara sobre celda oscura: el texto de la celda es la vista
            # de tabla y tiene que leerse en toda la rampa. Se decide por la
            # **luminancia real** del color y no por la posicion en la escala:
            # en una rampa divergente el centro es el tono mas claro, asi que
            # un valor alto en la escala puede tocar una celda palida y la
            # regla posicional pondria texto blanco sobre casi blanco.
            red, green, blue = cmap(norm(value))[:3]
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axes.text(
                column + 0.5,
                row + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=SURFACE if luminance < 0.5 else INK,
            )

    axes.set_xticks(np.arange(grid.shape[1]) + 0.5)
    axes.set_xticklabels([TYPE_LABELS.get(name, name) for name in grid.columns], fontsize=9)
    axes.set_yticks(np.arange(grid.shape[0]) + 0.5)
    axes.set_yticklabels(list(grid.index), fontsize=9)
    axes.tick_params(length=0, colors=INK_SECONDARY)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_title(title, fontsize=11, color=INK, pad=14, loc="left", fontweight="bold")
    # Coordenadas de ejes y no de datos: el eje y esta invertido para que la
    # primera fila quede arriba, asi que "debajo del panel" en datos seria y
    # creciente y dependeria del numero de filas.
    axes.text(
        0,
        -0.30,
        subtitle,
        fontsize=8.5,
        color=INK_MUTED,
        transform=axes.transAxes,
        clip_on=False,
    )
    return ScalarMappable(norm=norm, cmap=cmap)


def figure_heatmap(table: pd.DataFrame, path: Path) -> None:
    """Mapa de calor detector x tipo de anomalia, con dos lecturas del recall.

    Dos paneles porque las dos metricas discrepan a proposito y la discrepancia
    es el hallazgo: el recall por rangos exige solape y el de afiliacion premia
    la cercania. El de afiliacion se dibuja con escala **divergente centrada en
    0.5** justamente porque 0.5 es el valor del azar en esa metrica: con una
    rampa secuencial, "no mejor que aleatorio" se leeria como un tono medio
    respetable.

    Parameters
    ----------
    table
        Tabla larga de metricas.
    path
        Fichero PNG de salida.
    """
    import matplotlib.pyplot as plt

    ranged = _grid(table, "range_recall")
    affiliation = _grid(table, "affiliation_recall")

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 3.9), facecolor=SURFACE)
    figure.subplots_adjust(left=0.11, right=0.93, top=0.76, bottom=0.24, wspace=0.42)

    left = _heatmap_panel(
        axes[0],
        ranged,
        colours=SEQUENTIAL_BLUE,
        vmin=0.0,
        vmax=1.0,
        title="Recall por rangos",
        subtitle="Fraccion del evento cubierta, penalizando fragmentacion. 0 = no detectado.",
    )
    right = _heatmap_panel(
        axes[1],
        affiliation,
        colours=DIVERGING_RED_BLUE,
        vmin=0.0,
        vmax=1.0,
        title="Recall de afiliacion",
        subtitle="Cercania frente al azar. 0.50 = aleatorio (gris), no 0.",
    )

    for axis, mappable, ticks in (
        (axes[0], left, [0.0, 0.5, 1.0]),
        (axes[1], right, [0.0, 0.5, 1.0]),
    ):
        bar = figure.colorbar(mappable, ax=axis, fraction=0.035, pad=0.02, ticks=ticks)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, labelsize=8, colors=INK_MUTED)

    figure.suptitle(
        "Que tipo de anomalia captura cada detector",
        fontsize=13,
        color=INK,
        x=0.012,
        ha="left",
        y=0.96,
        fontweight="bold",
    )
    figure.text(
        0.012,
        0.885,
        f"3 series x {ROUNDS} eventos por tipo · alfa={ALPHA} · modelo base MSTL · "
        "agregado juntando rangos, no promediando series",
        fontsize=9,
        color=INK_SECONDARY,
        ha="left",
    )
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    print(f"  {path}")


def _macro_pr_curve(
    scores: pd.DataFrame, truth_index: set[tuple[str, pd.Timestamp]], grid: np.ndarray
) -> np.ndarray:
    """Curva PR promediada entre series sobre una rejilla comun de recall.

    Se promedia entre series y no se agrupan las puntuaciones en un unico
    ranking porque `FittedDetector.score` declara que el score es ordinal solo
    **dentro** de un par (detector, serie). Agruparlas dibujaria una curva que
    ningun umbral produce.

    Parameters
    ----------
    scores
        Scores de un detector, ya restringidos al soporte comun.
    truth_index
        Claves ``(unique_id, ds)`` anomalas.
    grid
        Rejilla de recall sobre la que interpolar.

    Returns
    -------
    numpy.ndarray
        Precision media en cada punto de `grid`.
    """
    from chronolab.evaluation.anomaly_metrics import pr_curve

    curves: list[np.ndarray] = []
    for uid, group in scores.groupby("unique_id", sort=True):
        ordered = group.sort_values("ds")
        labels = np.array([(str(uid), ts) in truth_index for ts in ordered["ds"]], dtype=bool)
        if not labels.any():
            continue
        precision, recall, _ = pr_curve(ordered["score"].to_numpy(dtype=float), labels)
        # Escalonada: la precision alcanzable a un recall dado es la del ultimo
        # umbral que lo alcanza, no una interpolacion lineal (que en el espacio
        # PR es incorrecta y sesga al alza).
        curves.append(
            precision[np.searchsorted(recall, grid, side="left").clip(0, recall.size - 1)]
        )
    return np.mean(curves, axis=0) if curves else np.full(grid.size, np.nan)


def figure_pr_curves(
    scores: pd.DataFrame, truth: pd.DataFrame, table: pd.DataFrame, path: Path
) -> None:
    """Curvas precision-recall puntuales, una por detector.

    Es la referencia minima del modulo de metricas: la curva puntual clasica,
    sin ninguna nocion de rango. Se dibuja con la linea base de prevalencia
    porque el suelo de una curva PR **es** la prevalencia, y una curva sin ella
    no se puede leer.

    Parameters
    ----------
    scores
        Scores apilados de los cuatro detectores, con `in_support`.
    truth
        Tabla de verdad.
    table
        Tabla larga de metricas, de la que se toma el AUC-PR ya calculado.
    path
        Fichero PNG de salida.
    """
    import matplotlib.pyplot as plt

    truth_index = {
        (str(uid), pd.Timestamp(ts))
        for uid, ts in zip(truth["unique_id"], truth["ds"], strict=True)
    }
    grid = np.linspace(0.0, 1.0, 201)
    pooled = table.loc[table["unique_id"].isna() & (table["anomaly_type"] == "all")]
    areas = pooled.loc[pooled["metric"] == "auc_pr"].set_index("detector_id")["value"]
    baseline = float(pooled.loc[pooled["metric"] == "auc_pr_baseline", "value"].mean())

    figure, axes = plt.subplots(figsize=(8.2, 5.4), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    figure.subplots_adjust(left=0.10, right=0.72, top=0.80, bottom=0.12)

    usable = scores.loc[scores["in_support"].fillna(False).astype(bool)]
    # Se ordena por el nombre corto, no partiendo el identificador por "_":
    # `lstm_ae_...` y `matrix_profile_...` llevan guion bajo dentro del propio
    # prefijo, asi que trocear daria "lstm" y "matrix", que no son claves.
    display_order = list(DISPLAY_NAMES.values())
    ordered_ids = sorted(
        usable["detector_id"].unique(),
        key=lambda name: (
            display_order.index(short_name(str(name)))
            if short_name(str(name)) in display_order
            else len(display_order)
        ),
    )
    # Color por identidad del detector, fijo en `DISPLAY_NAMES`, no por su
    # puesto en el ranking de esta ejecucion.
    palette = {
        name: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, name in enumerate(DISPLAY_NAMES.values())
    }

    axes.axhline(baseline, color=AXIS, linewidth=1.0, zorder=1)
    axes.text(
        0.015,
        baseline + 0.022,
        f"prevalencia {baseline:.3f} — el suelo de cualquier curva PR",
        fontsize=8.5,
        color=INK_MUTED,
        va="bottom",
        ha="left",
    )

    handles = []
    for detector_id in ordered_ids:
        name = short_name(str(detector_id))
        precision = _macro_pr_curve(
            usable.loc[usable["detector_id"] == detector_id], truth_index, grid
        )
        area = float(areas.get(detector_id, np.nan))
        line = axes.plot(
            grid,
            precision,
            color=palette[name],
            linewidth=2.0,
            zorder=3,
            label=f"{name}\nAUC-PR {area:.3f}",
        )
        handles.append(line[0])

    # Leyenda a la derecha y no etiquetas pegadas a cada curva: las cuatro
    # convergen en el extremo de recall alto, asi que ahi las etiquetas se
    # pisan unas a otras. El texto de la leyenda lleva el nombre y el area, de
    # modo que la identidad sigue sin depender solo del color.
    legend = axes.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
        labelspacing=1.1,
        handlelength=1.6,
        borderaxespad=0.0,
    )
    for text, handle in zip(legend.get_texts(), handles, strict=True):
        text.set_color(handle.get_color())

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    axes.set_xlabel("Recall puntual", fontsize=9.5, color=INK_SECONDARY)
    axes.set_ylabel("Precision puntual", fontsize=9.5, color=INK_SECONDARY)
    axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(labelsize=8.5, colors=INK_MUTED, length=0)

    figure.suptitle(
        "Curvas precision-recall puntuales por detector",
        fontsize=13,
        color=INK,
        x=0.012,
        ha="left",
        y=0.965,
        fontweight="bold",
    )
    figure.text(
        0.012,
        0.885,
        "La referencia minima: sin nocion de rango, cuenta instantes. Promediada entre las 3\n"
        "series, no agrupada: el score solo es ordinal dentro de un par (detector, serie).",
        fontsize=9,
        color=INK_SECONDARY,
        ha="left",
    )
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    print(f"  {path}")


def draw_figures() -> None:
    """Redibuja las figuras desde los artefactos ya persistidos."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(RESULTS_PATH)
    scores = pd.read_parquet(SCORES_PATH)
    truth = pd.read_parquet(TRUTH_PATH)
    print("figuras:")
    figure_heatmap(table, FIGURES_DIR / "09_anomaly_heatmap.png")
    figure_pr_curves(scores, truth, table, FIGURES_DIR / "09_anomaly_pr_curves.png")


def main() -> None:
    """Ejecuta el hito completo y persiste los artefactos."""
    if "--figures-only" in sys.argv:
        draw_figures()
        summarise(pd.read_parquet(RESULTS_PATH))
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    panel, truth, plan = build_contaminated_panel()
    result = run_backtest(panel, plan)
    scores = score_detectors(result)
    table, support = evaluate(scores, truth)

    table.to_parquet(RESULTS_PATH, index=False)
    stacked = pd.concat(
        [frame.assign(detector_id=str(detector_id)) for detector_id, frame in scores.items()],
        ignore_index=True,
    )
    stacked = stacked.merge(
        support.rename(columns={"scorable": "in_support"}), on=["unique_id", "ds"], how="left"
    )
    stacked.to_parquet(SCORES_PATH, index=False)
    truth.to_parquet(TRUTH_PATH, index=False)

    print(f"\nescritos:\n  {RESULTS_PATH}\n  {SCORES_PATH}\n  {TRUTH_PATH}")
    draw_figures()
    summarise(table)


if __name__ == "__main__":
    main()
