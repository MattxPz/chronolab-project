"""Backtest completo del hito deep: NHITS, TFT, PatchTST y el LSTM propio.

Ejecuta el hito de punta a punta sobre el mismo panel horario sintetico y el
mismo plan de backtesting que `scripts/run_ml_feature_analysis.py`, de modo que
los modelos profundos caigan en la misma tabla, sobre las mismas ventanas de
holdout, que los baselines, los estadisticos y los de gradient boosting ya
publicados:

1. Corre el backtest completo (19 modelos) y actualiza el leaderboard, ahora
   con la columna `n_params` junto a los tiempos de ajuste e inferencia.
2. Dibuja el eje **precision-coste**: MASE frente a segundos de ajuste y
   frente a numero de parametros.
3. Extrae del TFT los pesos de seleccion de variables y la atencion temporal,
   los persiste en `reports/results/tft_interpretability.parquet` —en el
   formato largo de la tabla `explanations` de docs/ARCHITECTURE.md §7.4, para
   que la pagina de explicabilidad de la app los lea sin recalcular nada— y
   guarda sus figuras.
4. Escribe los hallazgos en `docs/DEEP_ANALYSIS.md`.

Uso: ``uv run --extra ml --extra deep python scripts/run_deep_analysis.py``

`refit_every=1` es obligatorio en este plan y no una preferencia: tanto las
redes de neuralforecast como la cabeza del LSTM propio emiten exactamente los
`h` pasos siguientes a su cutoff, asi que reutilizar un ajuste en una ventana
posterior desalinearia la prediccion. Los adaptadores lo detectan y fallan
ruidosamente; el plan lo evita de raiz pagando un ajuste por ventana.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tests.fixtures.synthetic import make_hourly_panel  # noqa: E402

from chronolab.data.futr import RealizedFutrProvider  # noqa: E402
from chronolab.errors import PerfectForesightWarning  # noqa: E402
from chronolab.evaluation.aggregate import build_leaderboard  # noqa: E402
from chronolab.evaluation.backtest import BacktestPlan, backtest  # noqa: E402
from chronolab.models.adapters.mlforecast import (  # noqa: E402
    LightGBMForecaster,
    XGBoostForecaster,
)
from chronolab.models.adapters.neuralforecast import (  # noqa: E402
    NHITSForecaster,
    PatchTSTForecaster,
    TFTForecaster,
    quiet_lightning,
)
from chronolab.models.adapters.prophet import ProphetForecaster  # noqa: E402
from chronolab.models.adapters.statsforecast import (  # noqa: E402
    AutoARIMAForecaster,
    AutoETSForecaster,
    AutoThetaForecaster,
    MSTLForecaster,
)
from chronolab.models.adapters.torch_lstm import LSTMForecaster  # noqa: E402
from chronolab.models.baselines import (  # noqa: E402
    DriftForecaster,
    HistoricAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    WindowAverageForecaster,
)
from chronolab.models.protocols import Forecaster  # noqa: E402
from chronolab.models.torch.trainer import TrainConfig  # noqa: E402
from chronolab.panel import Panel  # noqa: E402
from chronolab.types import ModelId  # noqa: E402

FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports" / "results"
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.parquet"
INTERPRETABILITY_PATH = RESULTS_DIR / "tft_interpretability.parquet"
DOC_PATH = ROOT / "docs" / "DEEP_ANALYSIS.md"

H = 24
N_WINDOWS = 6
HOLDOUT_WINDOWS = 2
SEED = 0
INPUT_SIZE = 168

DEEP_MODEL_IDS = ("nhits", "tft", "patchtst", "lstm")

# Hiperparametros tuneados en el hito anterior (`scripts/run_ml_feature_analysis.py`,
# Optuna sobre las ventanas dev). Se reutilizan tal cual para que los modelos de
# gradient boosting sigan siendo exactamente los mismos que ya estan publicados.
LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 250,
    "learning_rate": 0.048046413497514796,
    "num_leaves": 51,
}
XGB_PARAMS: dict[str, object] = {
    "n_estimators": 150,
    "learning_rate": 0.20761410420015303,
    "max_depth": 8,
}


def classical_models() -> list[Forecaster]:
    """Baselines, estadisticos y gradient boosting: el estado del leaderboard antes de este hito."""
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(season=24),
        SeasonalNaiveForecaster(season=168, model_id=ModelId("seasonal_naive_168")),
        WindowAverageForecaster(),
        HistoricAverageForecaster(),
        DriftForecaster(),
        AutoARIMAForecaster(h=H, season_length=24, use_intervals=False),
        AutoETSForecaster(h=H, season_length=24, use_intervals=False),
        AutoThetaForecaster(h=H, season_length=24, use_intervals=False),
        MSTLForecaster(h=H, season_lengths=(24, 168), use_intervals=False),
        ProphetForecaster(),
        LightGBMForecaster(
            strategy="recursive",
            params=LGBM_PARAMS,
            seed=SEED,
            model_id=ModelId("lightgbm_recursive"),
        ),
        LightGBMForecaster(
            strategy="direct", params=LGBM_PARAMS, seed=SEED, model_id=ModelId("lightgbm_direct")
        ),
        XGBoostForecaster(
            strategy="recursive",
            params=XGB_PARAMS,
            seed=SEED,
            model_id=ModelId("xgboost_recursive"),
        ),
        XGBoostForecaster(
            strategy="direct", params=XGB_PARAMS, seed=SEED, model_id=ModelId("xgboost_direct")
        ),
    ]


def deep_models() -> list[Forecaster]:
    """Los cuatro modelos del hito: tres de neuralforecast y el LSTM propio.

    Los presupuestos estan **medidos**, no elegidos a ojo. Sobre este panel
    (3 series, ~2000 horas) y en la CPU de referencia, un ajuste cuesta
    aproximadamente: NHITS 14 s, TFT 13 s, PatchTST 4 s y el LSTM propio 70 s.
    Con `refit_every=1` y 6 ventanas, eso deja el bloque profundo en torno a
    los diez minutos, que es lo que hace viable el run completo en CPU modesta
    —el requisito del hito.

    El parametro que mas manda es `windows_batch_size`: con el valor por
    defecto de neuralforecast (1024) un solo ajuste de TFT no termina en una
    hora sobre esta maquina. No pretenden agotar la capacidad de las redes; la
    comparacion es honesta porque todas se miden sobre exactamente las mismas
    ventanas y el coste que pagan queda publicado junto a su error.
    """
    budget = {
        "input_size": INPUT_SIZE,
        "max_steps": 150,
        "val_check_steps": 50,
        "early_stop_patience_steps": 2,
        "windows_batch_size": 64,
        "seed": SEED,
    }
    return [
        NHITSForecaster(**budget),  # type: ignore[arg-type]
        TFTForecaster(hidden_size=32, n_head=4, **budget),  # type: ignore[arg-type]
        PatchTSTForecaster(use_futr_exog=False, **budget),  # type: ignore[arg-type]
        LSTMForecaster(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2,
            config=TrainConfig(max_epochs=12, batch_size=128, patience=3),
            seed=SEED,
        ),
    ]


def plot_accuracy_cost(overall: pd.DataFrame) -> None:
    """Dibuja el eje precision-coste: MASE frente a segundos y frente a parametros."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, column, xlabel, title in (
        (
            axes[0],
            "fit_seconds_mean",
            "Segundos por ajuste (media)",
            "Precision vs coste de ajuste",
        ),
        (axes[1], "n_params", "Parametros entrenables", "Precision vs tamano del modelo"),
    ):
        data = overall.dropna(subset=[column, "mase"])
        data = data[data[column] > 0]
        deep = data[data["model_id"].isin(DEEP_MODEL_IDS)]
        rest = data[~data["model_id"].isin(DEEP_MODEL_IDS)]

        ax.scatter(rest[column], rest["mase"], s=45, color="steelblue", label="resto")
        ax.scatter(deep[column], deep["mase"], s=70, color="crimson", marker="D", label="deep")
        for _, row in data.iterrows():
            ax.annotate(
                str(row["model_id"]),
                (row[column], row["mase"]),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MASE (holdout)")
        ax.set_title(title)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = FIGURES_DIR / "08_accuracy_vs_cost.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")


def plot_tft_interpretability(importance: pd.DataFrame, attention: pd.DataFrame) -> None:
    """Dibuja los pesos de seleccion de variables y la atencion temporal del TFT."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    past = importance[importance["block"] == "past"].sort_values("value")
    axes[0].barh(past["feature"], past["value"], color="teal")
    axes[0].set_xlabel("Peso medio de seleccion de variables")
    axes[0].set_title("TFT: seleccion de variables (bloque pasado)")

    axes[1].plot(attention["offset"], attention["value"], color="darkorange")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1, label="cutoff")
    axes[1].set_xlabel("Pasos relativos al cutoff")
    axes[1].set_ylabel("Atencion media recibida")
    axes[1].set_title("TFT: atencion temporal del horizonte")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = FIGURES_DIR / "08_tft_interpretability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada: {path}")


def extract_tft_interpretability(
    panel: Panel, plan: BacktestPlan
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajusta un TFT sobre la ultima ventana de holdout y extrae sus pesos.

    Se reajusta a proposito en vez de reutilizar el del backtest: el motor no
    devuelve los objetos ajustados —solo sus artefactos— y los pesos de
    interpretabilidad describen la ultima pasada hacia delante de la red, asi
    que hay que controlar sobre que ventana se hace esa pasada.
    """
    windows = plan.splitter().split(panel)
    window = windows[-1]
    train = panel.train(window)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PerfectForesightWarning)
        provider = RealizedFutrProvider(panel=panel)

    model = TFTForecaster(
        input_size=INPUT_SIZE,
        max_steps=200,
        val_check_steps=50,
        early_stop_patience_steps=3,
        hidden_size=32,
        n_head=4,
        seed=SEED,
    )
    fitted = model.fit(train, h=plan.h)
    fitted.predict(provider.futr(window, ids=train.ids()))  # fija los pesos sobre esta ventana

    return fitted.variable_importance(), fitted.temporal_attention()


def _markdown_table(df: pd.DataFrame, columns: list[str], *, float_format: str = "{:.4f}") -> str:
    """Tabla Markdown minima, sin depender de `tabulate` (no es dependencia del proyecto)."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("—")
            elif isinstance(value, float):
                cells.append(float_format.format(value))
            else:
                cells.append(f"{value:,}" if isinstance(value, int) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_findings_doc(
    *, leaderboard: pd.DataFrame, importance: pd.DataFrame, attention: pd.DataFrame
) -> None:
    """Escribe `docs/DEEP_ANALYSIS.md` con los hallazgos del run."""
    overall = (
        leaderboard[leaderboard["unique_id"].isna()].sort_values("mase").reset_index(drop=True)
    )
    columns = [
        "model_id",
        "mase",
        "mae",
        "rmse",
        "n_params",
        "fit_seconds_mean",
        "predict_seconds_mean",
    ]
    full_table = _markdown_table(overall, columns)

    deep = overall[overall["model_id"].isin(DEEP_MODEL_IDS)]
    deep_table = _markdown_table(deep, columns)

    best = overall.iloc[0]
    best_deep = deep.iloc[0]
    lstm = overall[overall["model_id"] == "lstm"].iloc[0]
    mstl = overall[overall["model_id"] == "mstl"].iloc[0]

    lstm_vs_mstl = (
        f"El LSTM propio queda en MASE {lstm['mase']:.4f} frente al {mstl['mase']:.4f} de MSTL: "
        + (
            "lo bate."
            if lstm["mase"] < mstl["mase"]
            else f"**pierde**, por {lstm['mase'] - mstl['mase']:+.4f}."
        )
    )

    past_table = _markdown_table(importance[importance["block"] == "past"], ["feature", "value"])
    future_table = _markdown_table(
        importance[importance["block"] == "future"], ["feature", "value"]
    )
    peak = attention.loc[attention["value"].idxmax()]
    context_share = float(attention.loc[attention["offset"] <= 0, "value"].sum())

    content = f"""# Análisis de los modelos profundos

Generado por `scripts/run_deep_analysis.py` sobre el panel horario sintético de
`tests.fixtures.synthetic` (3 series, ~2000 horas), con el mismo plan de
backtesting que el resto del leaderboard: {N_WINDOWS} ventanas de origen rodante
(h={H}, `step_size`={H}, `holdout_windows`={HOLDOUT_WINDOWS}) y `refit_every=1`.
Todos los números de este documento son trazables a ese run; ninguno está
escrito a mano.

## 1. Leaderboard completo (holdout, agregado sobre todas las series)

{full_table}

Las tres columnas de la derecha son el **eje precisión-coste**, que este hito
añade al leaderboard: `n_params` (nulo en los modelos que no ajustan parámetros
por optimización), segundos medios por ajuste y segundos medios por inferencia.

## 2. Los cuatro modelos del hito

{deep_table}

- Mejor modelo del leaderboard: **{best["model_id"]}** (MASE {best["mase"]:.4f}).
- Mejor modelo profundo: **{best_deep["model_id"]}** (MASE {best_deep["mase"]:.4f}).
- {lstm_vs_mstl}

### Precisión frente a coste

![Precisión vs coste](figures/08_accuracy_vs_cost.png)

El eje horizontal es logarítmico en las dos gráficas. Leerlas juntas es el
punto: un modelo que gana por poco pagando dos órdenes de magnitud más de
cómputo no es, en general, el que se despliega.

## 3. Honestidad sobre el LSTM propio

El objetivo declarado de la Parte B no era ganar, sino estar correctamente
implementado y honestamente evaluado. Lo que sostiene esa afirmación:

- **Escalado ajustado solo con train y revertido al predecir.** `SeriesScaler`
  se ajusta dentro de `fit`, sobre el `Panel` que el motor ya recortó a
  `ds <= cutoff`, y `inverse_target` devuelve las predicciones a la escala de
  cada serie. Hay tests que comprueban las dos mitades por separado.
- **Ventanas causales.** El contexto termina en `t` y el objetivo empieza en
  `t+1`; ninguna ventana incluye en la entrada el primer instante que predice.
  Un test de estabilidad por prefijos lo verifica al estilo del T1 de
  `docs/ARCHITECTURE.md`.
- **Early stopping que restaura los mejores pesos**, no los últimos —el error
  clásico que convierte el early stopping en ruido—, con gradient clipping y
  `ReduceLROnPlateau` sobre la misma señal de validación.
- **Reproducible**: semilla en Python, numpy, torch y el generador del
  `DataLoader`. Dos ajustes con la misma semilla dan la misma curva de pérdida.
- **Predicción directa multi-paso**: la cabeza proyecta los {H} pasos de una
  vez, así que el error no se acumula por realimentación. A cambio no puede
  reutilizarse en una ventana posterior, y el adaptador **falla ruidosamente**
  en vez de publicar una predicción desalineada.

Si el número de arriba dice que pierde contra MSTL, eso es el resultado. Un
LSTM de decenas de miles de parámetros entrenado con un presupuesto acotado
sobre tres series sintéticas de estacionalidad limpia no tiene por qué batir a
una descomposición estacional múltiple diseñada exactamente para ese caso.

## 4. Interpretabilidad del TFT

![Interpretabilidad del TFT](figures/08_tft_interpretability.png)

Pesos de selección de variables (*variable selection network*), promediados
sobre el tiempo y el lote. Suman uno dentro de cada bloque porque son la salida
de un softmax sobre las variables:

**Bloque pasado**

{past_table}

**Bloque futuro**

{future_table}

**Atención temporal**: el máximo está en el offset {int(peak["offset"])} respecto al
cutoff, y el {context_share:.1%} de la atención total la reciben instantes del
contexto (offset ≤ 0) frente al resto, que se la reparten los propios pasos del
horizonte.

Ambas tablas se persisten en `reports/results/tft_interpretability.parquet` en
el formato largo de la tabla `explanations` de `docs/ARCHITECTURE.md` §7.4
(`kind` en `attention_variable` / `attention_temporal`), de modo que la página
de explicabilidad de la app las lea y las dibuje sin recalcular nada (A5).

## 5. Metodología y limitaciones declaradas

- **Vintage de las exógenas futuras**: `RealizedFutrProvider`, es decir,
  presciencia perfecta. Todo el leaderboard es una cota superior de rendimiento,
  no una estimación de producción. Afecta especialmente al TFT y a NHITS, que
  son los que más peso dan a la temperatura.
- **`refit_every=1` obligatorio**: las redes emiten exactamente los `h` pasos
  siguientes a su cutoff. Reutilizar un ajuste desalinearía la predicción, y
  los adaptadores lo detectan y fallan en vez de publicarla.
- **Presupuestos acotados**: `max_steps` en neuralforecast y `max_epochs` en el
  LSTM propio están fijados para que el run completo sea viable en CPU. Subirlos
  es legítimo y el coste quedaría reflejado en las mismas columnas.
- **PatchTST es univariado**: neuralforecast no admite exógenas futuras en ese
  modelo. El adaptador lo rechaza en construcción en lugar de aceptarlas y
  descartarlas en silencio, así que compite sin temperatura y eso hay que
  tenerlo en cuenta al compararlo con NHITS y TFT.
"""
    DOC_PATH.write_text(content, encoding="utf-8")
    print(f"Hallazgos escritos en {DOC_PATH}")


def main() -> None:
    """Corre el hito deep completo: backtest, leaderboard, figuras e interpretabilidad."""
    quiet_lightning()
    warnings.filterwarnings("ignore", category=FutureWarning)

    panel = make_hourly_panel()
    # `refit_every=1` no es una preferencia aqui: ver el docstring del modulo.
    plan = BacktestPlan(
        h=H,
        n_windows=N_WINDOWS,
        step_size=H,
        holdout_windows=HOLDOUT_WINDOWS,
        refit_every=1,
        seed=SEED,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PerfectForesightWarning)
        futr = RealizedFutrProvider(panel=panel)

    models = [*classical_models(), *deep_models()]
    model_ids = [str(model.model_id) for model in models]
    assert len(set(model_ids)) == len(model_ids), f"model_id repetido: {model_ids}"

    print(f"== Backtest completo: {len(models)} modelos, {N_WINDOWS} ventanas ==")
    result = backtest(panel, models, plan, futr=futr)
    estados = result.model_runs.groupby("model_id")["status"].value_counts()
    print(estados.to_string())

    leaderboard = build_leaderboard(result, panel, stage="holdout", path=LEADERBOARD_PATH)
    print(f"\nLeaderboard actualizado: {LEADERBOARD_PATH} ({len(leaderboard)} filas)")

    overall = leaderboard[leaderboard["unique_id"].isna()].sort_values("mase")
    print(
        overall[
            ["model_id", "mase", "n_params", "fit_seconds_mean", "predict_seconds_mean"]
        ].to_string(index=False)
    )

    print("\n== Eje precision-coste ==")
    plot_accuracy_cost(overall)

    print("\n== Interpretabilidad del TFT ==")
    importance, attention = extract_tft_interpretability(panel, plan)
    plot_tft_interpretability(importance, attention)

    persisted: pd.DataFrame = pd.concat(
        [
            importance.assign(ds=pd.NaT, offset=pd.NA),
            attention.assign(feature=pd.NA, block=pd.NA),
        ],
        ignore_index=True,
    )[["kind", "feature", "block", "ds", "offset", "value"]]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    persisted.to_parquet(INTERPRETABILITY_PATH, index=False)
    print(f"Interpretabilidad persistida: {INTERPRETABILITY_PATH} ({len(persisted)} filas)")

    write_findings_doc(leaderboard=leaderboard, importance=importance, attention=attention)
    print("\nListo.")


if __name__ == "__main__":
    main()
