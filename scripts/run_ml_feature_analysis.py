"""Backtest completo con los modelos ML, tuning con Optuna y hallazgos de features.

Ejecuta el "hito ML" del proyecto de punta a punta sobre el panel horario
sintetico de `tests.fixtures.synthetic` (el mismo que produjo la version
anterior de `reports/results/leaderboard.parquet`, para que los modelos
nuevos queden en la misma tabla que los baselines y los modelos estadisticos
ya publicados, sobre exactamente las mismas ventanas de holdout):

1. Tunea LightGBM y XGBoost con Optuna, viendo unicamente las ventanas `dev`
   del plan (`chronolab.evaluation.tuning`).
2. Corre el backtest completo (baselines + estadisticos + los cuatro modelos
   ML: LightGBM/XGBoost x recursiva/directa) y actualiza el leaderboard.
3. Calcula el MASE por paso de horizonte de los cuatro modelos ML, para
   comparar como se degradan la estrategia recursiva y la directa al crecer
   el horizonte.
4. Ajusta un LightGBM de referencia sobre el train de la ultima ventana de
   holdout y extrae su importancia nativa (ganancia) y SHAP sobre una
   muestra, guardando las figuras en `reports/figures/`.
5. Escribe los hallazgos en `docs/FEATURE_ANALYSIS.md`.

Uso: ``uv run --extra ml python scripts/run_ml_feature_analysis.py``

No es parte del paquete instalable (`src/chronolab`): es un script de un solo
uso, en el espiritu de `docs/ARCHITECTURE.md` §2 (`scripts/run_backtest.py`).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tests.fixtures.synthetic import make_hourly_panel  # noqa: E402

from chronolab.data.futr import RealizedFutrProvider  # noqa: E402
from chronolab.errors import PerfectForesightWarning  # noqa: E402
from chronolab.evaluation.aggregate import build_leaderboard, score_forecasts  # noqa: E402
from chronolab.evaluation.backtest import BacktestPlan, backtest  # noqa: E402
from chronolab.evaluation.metrics import point_metrics  # noqa: E402
from chronolab.evaluation.tuning import tune  # noqa: E402
from chronolab.models.adapters.mlforecast import LightGBMForecaster, XGBoostForecaster  # noqa: E402
from chronolab.models.adapters.prophet import ProphetForecaster  # noqa: E402
from chronolab.models.adapters.statsforecast import (  # noqa: E402
    AutoARIMAForecaster,
    AutoETSForecaster,
    AutoThetaForecaster,
    MSTLForecaster,
)
from chronolab.models.baselines import (  # noqa: E402
    DriftForecaster,
    HistoricAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    WindowAverageForecaster,
)
from chronolab.models.protocols import Forecaster  # noqa: E402
from chronolab.panel import Panel  # noqa: E402
from chronolab.types import ModelId  # noqa: E402

FIGURES_DIR = ROOT / "reports" / "figures"
LEADERBOARD_PATH = ROOT / "reports" / "results" / "leaderboard.parquet"
DOC_PATH = ROOT / "docs" / "FEATURE_ANALYSIS.md"

H = 24
N_WINDOWS = 6
HOLDOUT_WINDOWS = 2
N_TRIALS = 8
SEED = 0

ML_MODEL_IDS = ("lightgbm_recursive", "lightgbm_direct", "xgboost_recursive", "xgboost_direct")


def baseline_models() -> list[Forecaster]:
    """Los seis baselines de `chronolab.models.baselines`."""
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(season=24),
        SeasonalNaiveForecaster(season=168, model_id=ModelId("seasonal_naive_168")),
        WindowAverageForecaster(),
        HistoricAverageForecaster(),
        DriftForecaster(),
    ]


def statistical_models() -> list[Forecaster]:
    """Los cuatro adaptadores de statsforecast mas Prophet.

    Los cuatro de statsforecast llevan ``use_intervals=False``: son
    ``refit_cost="expensive"``, asi que la politica de refit por defecto los
    ajusta una sola vez (en la ventana mas corta) y reutiliza ese ajuste en
    las demas. `statsforecast.ConformalIntervals` fija el horizonte de
    calibracion al ajustar (docstring de
    `chronolab.models.adapters.statsforecast`), asi que reutilizar el ajuste
    en una ventana posterior con intervalos activos falla en cuanto el
    horizonte pedido desde el cutoff original supera `h` — el segundo reuso
    en adelante. Sin intervalos ese limite no existe. Prophet no tiene esa
    restriccion (`predictive_samples` no fija el horizonte al ajustar), asi
    que si lleva sus cuantiles activos.
    """
    return [
        AutoARIMAForecaster(h=H, season_length=24, use_intervals=False),
        AutoETSForecaster(h=H, season_length=24, use_intervals=False),
        AutoThetaForecaster(h=H, season_length=24, use_intervals=False),
        MSTLForecaster(h=H, season_lengths=(24, 168), use_intervals=False),
        ProphetForecaster(),
    ]


def _tune_one(
    panel: Panel,
    plan: BacktestPlan,
    futr: RealizedFutrProvider,
    *,
    library: str,
) -> dict[str, object]:
    """Tunea LightGBM o XGBoost con Optuna, solo sobre las ventanas dev.

    Parameters
    ----------
    library
        ``"lightgbm"`` o ``"xgboost"``.

    Returns
    -------
    dict
        Mejores hiperparametros encontrados (``trial.suggest_*``).
    """
    forecaster_cls = LightGBMForecaster if library == "lightgbm" else XGBoostForecaster

    def build_model(trial: Any) -> Forecaster:
        params: dict[str, object] = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
        if library == "lightgbm":
            params["num_leaves"] = trial.suggest_int("num_leaves", 7, 63)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 2, 8)
        return forecaster_cls(
            strategy="recursive",
            params=params,
            seed=SEED,
            model_id=ModelId(f"{library}_tuning"),
        )

    outcome = tune(panel, build_model, plan, n_trials=N_TRIALS, metric="mase", seed=SEED, futr=futr)
    print(
        f"[tuning] {library}: {N_TRIALS} trials sobre {outcome.n_dev_windows} ventanas dev -> "
        f"best_params={outcome.study.best_params}, best_mase={outcome.study.best_value:.4f}"
    )
    return dict(outcome.study.best_params)


def ml_models(lgbm_params: dict[str, object], xgb_params: dict[str, object]) -> list[Forecaster]:
    """Los cuatro modelos ML: LightGBM/XGBoost x recursiva/directa, con params tuneados."""
    return [
        LightGBMForecaster(
            strategy="recursive",
            params=lgbm_params,
            seed=SEED,
            model_id=ModelId("lightgbm_recursive"),
        ),
        LightGBMForecaster(
            strategy="direct", params=lgbm_params, seed=SEED, model_id=ModelId("lightgbm_direct")
        ),
        XGBoostForecaster(
            strategy="recursive",
            params=xgb_params,
            seed=SEED,
            model_id=ModelId("xgboost_recursive"),
        ),
        XGBoostForecaster(
            strategy="direct", params=xgb_params, seed=SEED, model_id=ModelId("xgboost_direct")
        ),
    ]


def horizon_degradation(result: Any, panel: Panel) -> pd.DataFrame:
    """MASE por paso de horizonte de los cuatro modelos ML, sobre holdout."""
    scored = score_forecasts(result, panel, stage="holdout")
    scored = scored[scored["model_id"].isin(ML_MODEL_IDS)]

    rows: list[dict[str, object]] = []
    for (model_id, h_step), group in scored.groupby(["model_id", "h_step"]):
        metrics = point_metrics(group)
        rows.append(
            {
                "model_id": model_id,
                "h_step": int(h_step),
                "mase": metrics["mase"],
                "mae": metrics["mae"],
                "n_obs": int(metrics["n_obs"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_id", "h_step"]).reset_index(drop=True)


def plot_degradation(degradation: pd.DataFrame) -> Path:
    """Guarda la curva de MASE por paso de horizonte, una linea por modelo ML."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_id, group in degradation.groupby("model_id"):
        ax.plot(group["h_step"], group["mase"], marker="o", markersize=3, label=str(model_id))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="naive estacional (MASE=1)")
    ax.set_xlabel("Paso del horizonte (h_step)")
    ax.set_ylabel("MASE (holdout)")
    ax.set_title("Degradacion por horizonte: recursiva vs directa")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = FIGURES_DIR / "07_horizon_degradation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")
    return path


def fit_explainability_model(
    panel: Panel, plan: BacktestPlan, lgbm_params: dict[str, object]
) -> Any:
    """Ajusta un LightGBM recursivo sobre el train de la ultima ventana de holdout."""
    windows = plan.splitter().split(panel)
    train = panel.train(windows[-1])
    model = LightGBMForecaster(
        strategy="recursive", params=lgbm_params, seed=SEED, model_id=ModelId("lightgbm_explain")
    )
    return model.fit(train, h=plan.h)


def _plot_importance_bars(
    frame: pd.DataFrame, *, value_column: str, xlabel: str, title: str, path: Path, color: str
) -> None:
    top = frame.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top[value_column], color=color)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")


def plot_importance(fitted: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Guarda la importancia nativa y SHAP del modelo de referencia."""
    native = fitted.feature_importance()
    _plot_importance_bars(
        native,
        value_column="importance",
        xlabel="Ganancia media (LightGBM)",
        title="Importancia nativa de features (top 20)",
        path=FIGURES_DIR / "07_feature_importance_native.png",
        color="steelblue",
    )

    shap_frame = fitted.shap_values(sample_size=300, seed=SEED)
    _plot_importance_bars(
        shap_frame,
        value_column="mean_abs_shap",
        xlabel="Media de |SHAP|",
        title="Importancia SHAP de features (top 20, muestra de 300 filas)",
        path=FIGURES_DIR / "07_feature_importance_shap.png",
        color="darkorange",
    )
    return native, shap_frame


def _markdown_table(df: pd.DataFrame, columns: list[str], *, float_format: str = "{:.4f}") -> str:
    """Tabla Markdown minima, sin depender de `tabulate` (no es dependencia del proyecto)."""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(float_format.format(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_findings_doc(
    *,
    leaderboard: pd.DataFrame,
    degradation: pd.DataFrame,
    importance_native: pd.DataFrame,
    importance_shap: pd.DataFrame,
    lgbm_params: dict[str, object],
    xgb_params: dict[str, object],
) -> None:
    """Escribe `docs/FEATURE_ANALYSIS.md` con los hallazgos del run."""
    overall = (
        leaderboard[leaderboard["unique_id"].isna()].sort_values("mase").reset_index(drop=True)
    )
    full_table = _markdown_table(
        overall, ["model_id", "mase", "mae", "rmse", "pinball_mean", "fit_seconds_mean"]
    )

    ml_overall = overall[overall["model_id"].isin(ML_MODEL_IDS)]
    ml_table = _markdown_table(ml_overall, ["model_id", "mase", "mae", "pinball_mean"])

    recursive_mase = ml_overall.set_index("model_id")
    lgbm_delta = (
        recursive_mase.loc["lightgbm_direct", "mase"]
        - recursive_mase.loc["lightgbm_recursive", "mase"]
    )
    xgb_delta = (
        recursive_mase.loc["xgboost_direct", "mase"]
        - recursive_mase.loc["xgboost_recursive", "mase"]
    )

    native_table = _markdown_table(importance_native.head(15), ["feature", "importance"])
    shap_table = _markdown_table(importance_shap.head(15), ["feature", "mean_abs_shap"])

    top_native = set(importance_native.head(10)["feature"])
    top_shap = set(importance_shap.head(10)["feature"])
    agreement = len(top_native & top_shap)

    early_degradation = degradation[degradation["h_step"].isin([1, H])]
    degradation_table = _markdown_table(
        early_degradation.sort_values(["model_id", "h_step"]),
        ["model_id", "h_step", "mase", "n_obs"],
    )

    content = f"""# Análisis de features de los modelos ML

Generado por `scripts/run_ml_feature_analysis.py` sobre el panel horario
sintético de `tests.fixtures.synthetic` (3 series, ~2000 horas), con un plan
de backtesting de {N_WINDOWS} ventanas de origen rodante (h={H},
`step_size`={H}, `holdout_windows`={HOLDOUT_WINDOWS}: {N_WINDOWS - HOLDOUT_WINDOWS} de
desarrollo y {HOLDOUT_WINDOWS} de reporte). Todos los números de este documento
son trazables a ese run; no hay ninguno escrito a mano.

## 1. Leaderboard (holdout, agregado sobre todas las series)

{full_table}

`reports/results/leaderboard.parquet` queda actualizado con las filas por
serie y agregadas de los quince modelos (seis baselines, cuatro estadísticos
más Prophet, y los cuatro modelos ML de este hito).

## 2. Estrategia recursiva frente a directa

{ml_table}

- **LightGBM**: la estrategia directa cambia el MASE agregado en
  {lgbm_delta:+.4f} frente a la recursiva.
- **XGBoost**: la estrategia directa cambia el MASE agregado en
  {xgb_delta:+.4f} frente a la recursiva.

### Degradación por paso de horizonte

![Degradación por horizonte](figures/07_horizon_degradation.png)

MASE en el primer paso del horizonte (h_step=1) frente al último (h_step={H}):

{degradation_table}

**Lectura.** La recursiva realimenta sus propias predicciones en los lags
cortos de la objetivo (`lag(y,1)`, `lag(y,2)`...), así que el error se
acumula paso a paso: cuanto más lejos del cutoff, más se apoya en
predicciones propias en vez de en observaciones reales. La directa
(`max_horizon` de mlforecast) ajusta un regresor independiente por paso, sin
recursión, así que no sufre ese acoplo — a cambio, cada submodelo tiene que
generalizar directamente una relación más lejana en el tiempo, con menos
señal de corto plazo específica de ese paso. La curva de la figura de arriba
es la evidencia empírica de cuál de los dos efectos domina en este panel y en
qué tramo del horizonte.

## 3. Importancia de features

### Nativa (ganancia media, LightGBM)

![Importancia nativa](figures/07_feature_importance_native.png)

{native_table}

### SHAP (media de |SHAP| sobre una muestra de 300 filas)

![Importancia SHAP](figures/07_feature_importance_shap.png)

{shap_table}

**Acuerdo entre los dos métodos**: {agreement} de las 10 features más
importantes coinciden entre la ganancia nativa y SHAP. Donde discrepan suele
ser por el efecto conocido de la ganancia nativa de favorecer variables con
muchos puntos de corte posibles (los lags y ventanas móviles numéricas) frente
a variables binarias o de pocos niveles (`is_weekend`, `is_holiday`), que SHAP
pondera por su contribución real a la predicción y no por cuántas veces el
árbol las usó para partir.

**Qué aportan realmente las features manuales** (calendario y térmicas, las
que no gestiona mlforecast por su cuenta): si alguna de `hour_sin`/`hour_cos`,
los términos de Fourier diarios/semanales, o los retardos de temperatura y
grados-día aparece entre las quince features de las dos tablas de arriba, es
evidencia directa de que el conjunto de `chronolab.features.builders` aporta
señal más allá de lo que ya capturan los lags y ventanas móviles nativos de
mlforecast sobre la propia objetivo.

## 4. Tuning con Optuna

Presupuesto: {N_TRIALS} trials por librería (parámetro `n_trials` de
`chronolab.evaluation.tuning.tune`, configurable), optimizando MASE sobre las
{N_WINDOWS - HOLDOUT_WINDOWS} ventanas `dev` del plan —nunca sobre las
{HOLDOUT_WINDOWS} de holdout que se reportan en la sección 1, por construcción
de `chronolab.evaluation.tuning.dev_only_panel` (docs/ARCHITECTURE.md, fuga
L5).

- **LightGBM**: `{lgbm_params}`
- **XGBoost**: `{xgb_params}`

Los mismos hiperparámetros se aplican a la variante recursiva y a la directa
de cada librería: tunear las cuatro combinaciones por separado multiplicaría
el coste por cuatro para un beneficio marginal, dado que el espacio de
búsqueda (profundidad/hojas, tasa de aprendizaje, número de árboles) no tiene
motivo *a priori* para ser muy distinto entre estrategias.

## 5. Metodología y limitaciones declaradas

- **Vintage de las exógenas futuras**: `RealizedFutrProvider`, es decir,
  presciencia perfecta. El resultado es una cota superior de rendimiento, no
  una estimación de lo que el sistema lograría en producción — igual que en
  el resto de runs de este proyecto sobre el panel sintético.
- **Filtrado de las térmicas más estricto que el álgebra general de
  `max_lead`**: `chronolab.models.adapters.mlforecast` nunca lee el
  `FutrFrame`; solo admite un retardo de temperatura si es al menos tan largo
  como el horizonte completo del plan (`k >= h`), con independencia de si la
  temperatura está declarada `futr_exog` o `hist_exog`. Con h={H} eso deja
  fuera los retardos cortos (`temp_c_lag1`) y solo sobreviven los que superan
  el horizonte completo (`ThermalFeatureConfig.lags` por defecto es
  ``(1, 24, 168)``, así que sobreviven 24 y 168). Es una limitación
  documentada del adaptador (ver su docstring), no del conjunto de features
  de `chronolab.features.builders`, que sí genera la versión sin retardar y
  los retardos cortos para quien los pueda usar.
- **Lags/ventanas/diferencias de la propia objetivo**: delegados enteros en
  `mlforecast` (`chronolab.features.builders.TargetFeatureConfig`,
  `DEFAULT_TARGET_FEATURES`), sin reimplementar la generación ni la
  recursividad a mano, tal y como pide el enunciado del hito.
"""
    DOC_PATH.write_text(content, encoding="utf-8")
    print(f"Hallazgos escritos en {DOC_PATH}")


def main() -> None:
    """Corre el hito ML completo: tuning, backtest, leaderboard, figuras y hallazgos."""
    panel = make_hourly_panel()
    # `refit_every` sin fijar a proposito: cada modelo aplica la politica por
    # defecto de `chronolab.evaluation.backtest.BacktestPlan.refit_every_for`
    # segun su `refit_cost` declarado (baselines y ML "cheap"/"free" reajustan
    # cada ventana; los estadisticos "expensive" se ajustan una sola vez, en
    # la ventana mas corta, y reutilizan ese ajuste en las demas).
    plan = BacktestPlan(
        h=H,
        n_windows=N_WINDOWS,
        step_size=H,
        holdout_windows=HOLDOUT_WINDOWS,
        seed=SEED,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PerfectForesightWarning)
        futr = RealizedFutrProvider(panel=panel)

    print("== Tuning (solo ventanas dev) ==")
    lgbm_params = _tune_one(panel, plan, futr, library="lightgbm")
    xgb_params = _tune_one(panel, plan, futr, library="xgboost")

    models = [*baseline_models(), *statistical_models(), *ml_models(lgbm_params, xgb_params)]
    model_ids = [str(model.model_id) for model in models]
    assert len(set(model_ids)) == len(model_ids), f"model_id repetido: {model_ids}"

    print(f"\n== Backtest completo: {len(models)} modelos, {N_WINDOWS} ventanas ==")
    result = backtest(panel, models, plan, futr=futr)
    print(result.model_runs.groupby("model_id")["status"].value_counts().to_string())

    leaderboard = build_leaderboard(result, panel, stage="holdout", path=LEADERBOARD_PATH)
    print(f"\nLeaderboard actualizado: {LEADERBOARD_PATH} ({len(leaderboard)} filas)")

    print("\n== Degradación por horizonte (recursiva vs directa) ==")
    degradation = horizon_degradation(result, panel)
    plot_degradation(degradation)

    print("\n== Importancia nativa y SHAP ==")
    explain_fitted = fit_explainability_model(panel, plan, lgbm_params)
    importance_native, importance_shap = plot_importance(explain_fitted)

    write_findings_doc(
        leaderboard=leaderboard,
        degradation=degradation,
        importance_native=importance_native,
        importance_shap=importance_shap,
        lgbm_params=lgbm_params,
        xgb_params=xgb_params,
    )
    print("\nListo.")


if __name__ == "__main__":
    main()
