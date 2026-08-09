# chronolab

Plataforma de forecasting y detección de anomalías en series temporales, con
backtesting de origen rodante y barreras estructurales contra la fuga de
información temporal.

[![CI](https://github.com/MattxPz/chronolab-project/actions/workflows/ci.yml/badge.svg)](https://github.com/MattxPz/chronolab-project/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

**Demo desplegada:** _pendiente — `make app` la levanta en local en `localhost:8501`
(ver [Instalación y uso](#instalación-y-uso))._
<!-- TODO: enlace a Streamlit Community Cloud una vez desplegada -->

---

<!-- TODO: GIF o captura del dashboard. Páginas sugeridas para el GIF, en este
     orden: Overview (serie + calidad + MSTL) → Forecast (bandas de
     incertidumbre) → Leaderboard (tabla ordenable) → Anomalías (heatmap +
     curvas PR). Generar con `make app` y grabar con ScreenToGif/Peek. -->
![Dashboard de Chronolab](docs/assets/dashboard.gif)

---

## Resultados (backtest de referencia)

Panel horario sintético de 3 series (`s00`–`s02`), estacionalidad diaria y
semanal (`tests.fixtures.synthetic.make_hourly_panel`, semilla fija).
Backtest de origen rodante: 6 ventanas, 2 de holdout, `h=24`, reajuste por
ventana. Los números de abajo son el bloque de **holdout** (144 observaciones,
3 series × 24 pasos × 2 ventanas), tal como está persistido en
[`reports/results/leaderboard.parquet`](reports/results/leaderboard.parquet).

| # | Modelo | MASE | MAE | RMSE | Parámetros | Ajuste medio |
|--:|---|--:|--:|--:|--:|--:|
| 1 | `prophet` † | **0.279** | 0.651 | 0.798 | — | 1.64 s |
| 2 | `nhits` † | 0.284 | 0.663 | 0.820 | 301 749 | 3.27 s |
| 3 | `mstl` | 0.288 | 0.671 | 0.851 | — | 3.44 s |
| 4 | `lightgbm_direct` | 0.352 | 0.821 | 1.035 | — | 46.65 s |
| 5 | `lightgbm_recursive` | 0.353 | 0.822 | 1.006 | — | 4.08 s |
| 6 | `patchtst` | 0.381 | 0.888 | 1.094 | 220 136 | 6.67 s |
| 7 | `xgboost_recursive` | 0.385 | 0.897 | 1.107 | — | 2.42 s |
| 8 | `xgboost_direct` | 0.389 | 0.906 | 1.143 | — | 23.40 s |
| 9 | `seasonal_naive_168` | 0.426 | 0.992 | 1.214 | — | 0.001 s |
| 10 | `lstm` † | 0.428 | 0.997 | 1.209 | 67 304 | 55.76 s |
| 11 | `auto_theta` | 0.507 | 1.180 | 1.523 | — | 13.96 s |
| 12 | `tft` † | 0.622 | 1.450 | 1.781 | 56 468 | 33.80 s |
| 13 | `auto_ets` | 0.653 | 1.523 | 1.815 | — | 2.88 s |
| 14 | `seasonal_naive` (24) | 0.727 | 1.693 | 2.059 | — | 0.001 s |
| 15 | `auto_arima` | 0.766 | 1.784 | 2.158 | — | 5.18 s |
| 16 | `historic_average` | 2.396 | 5.586 | 6.356 | — | 0.001 s |
| 17 | `drift` | 2.444 | 5.697 | 6.341 | — | 0.001 s |
| 18 | `naive` | 2.445 | 5.699 | 6.343 | — | 0.001 s |
| 19 | `window_average` | 2.454 | 5.722 | 6.421 | — | 0.001 s |

`MASE < 1` bate al naive estacional (fila 14) sobre la escala de su propio
entrenamiento; los cuatro baselines de las últimas filas (14→19: naive,
drift, promedio histórico, promedio de ventana) no tienen ninguna estructura
que explotar y sirven de piso, no de referencia competitiva.

**† — recibieron la temperatura futura real, no una previsión.** `prophet`,
`nhits`, `tft` y `lstm` consumen `temp_c` como exógena conocida a futuro; en
este run esa columna se sirve con el valor **realizado** (presciencia
perfecta), no con un pronóstico degradado. Sus tres primeros puestos están
inflados por esa ventaja y no son comparables sin matiz con el resto — ver
[Limitaciones](#limitaciones-y-qué-haría-diferente).

Tabla completa (con métricas probabilísticas, cobertura y desglose por
serie) en el propio parquet o en `docs/DEEP_ANALYSIS.md` /
`docs/FEATURE_ANALYSIS.md`.

---

## Hallazgos

- **La estructura estacional es real y explotable.** Los cinco baselines sin
  ajuste (filas 14–19) quedan en MASE 0.73–2.45; los nueve primeros modelos
  bajan de 0.43. La brecha entre "repetir el ciclo anterior" y "modelar algo"
  es de casi un orden de magnitud en este panel — consistente con lo que la
  EDA ya mostraba: fuerza estacional 0.87–0.90 en `residential_north` y
  `commercial_mixed` una vez tratados los atípicos (`docs/EDA_FINDINGS.md` §6).

- **Un naive semanal (`seasonal_naive_168`, MASE 0.426) bate a `auto_arima`
  (0.766), `auto_ets` (0.653), `auto_theta` (0.507) y a `tft` (0.622).**
  Ninguno de esos tres modelos estadísticos clásicos modela la estacionalidad
  semanal explícitamente (frente a MSTL, que sí, y encabeza junto con
  `prophet`/`nhits`); el TFT paga 33.8 s de ajuste y 56 468 parámetros para
  perder contra copiar el valor de hace 7 días. Complejidad no es lo mismo
  que ajuste al problema.

- **El zero-shot (Chronos-Bolt) está implementado pero no en este leaderboard
  compartido.** `chronolab.models.adapters.chronos.ChronosForecaster` envuelve
  `amazon/chronos-bolt-small` sin ningún ajuste local y ya tiene tests y
  adaptador funcionando; el run que produjo la tabla de arriba
  (`scripts/run_deep_analysis.py`) no lo incluyó. Falta correrlo sobre el
  mismo panel y las mismas ventanas antes de poder decir si compite con los
  modelos ajustados — es la siguiente tarea, no un resultado.

- **La métrica de anomalías que se mire cambia quién gana, y una de ellas
  esconde un problema real.** Sobre 4 detectores × 6 tipos de anomalía
  (`docs/ANOMALY_FINDINGS.md`), el ganador depende de si se mide `range_f1`
  (Conformal, 0.456), `auc_pr` (IsolationForest, 0.571) o `range_recall`
  (LSTM-Autoencoder, 0.900). Ese último número es engañoso: IsolationForest y
  el autoencoder marcan el **48.6 %** y el **53.5 %** de los instantes
  evaluados al umbral nominal — su recall alto es cobertura por saturación,
  no detección, y solo se ve mirando la tasa de marcado junto al recall
  (`false_alarms_per_1000` = 37.3 y su `range_precision` = 0.140 lo confirman).

---

## Cómo está construido

```mermaid
flowchart LR
    subgraph Ingesta
        RAW[Fuentes: REE, Open-Meteo,<br/>UCI, Binance, sintético]
    end
    subgraph Contrato
        PANEL[Panel<br/>rejilla UTC completa,<br/>roles target/futr/hist declarados]
    end
    subgraph Backtesting
        SPLIT[RollingOriginSplitter<br/>único emisor de ventanas]
        LOOP[Motor: por ventana,<br/>fit(train ≤ cutoff) → predict]
    end
    subgraph Evaluación
        METRICS[MASE / RMSE / pinball / CRPS<br/>por serie y ventana]
        DM[Diebold-Mariano]
        LB[leaderboard.parquet]
    end
    subgraph Anomalías
        RESID[Residuos de un modelo base<br/>MSTL]
        DETECT[4 detectores: Conformal,<br/>IsolationForest, LSTM-AE, MatrixProfile]
        ANOM_METRICS[Range F1 / afiliación /<br/>VUS-PR / operativas]
    end
    subgraph Presentación
        APP[Streamlit: Overview,<br/>Forecast, Leaderboard, Anomalías]
        API[FastAPI de solo lectura]
    end

    RAW --> PANEL --> SPLIT --> LOOP --> METRICS --> LB
    METRICS --> DM
    PANEL --> RESID --> DETECT --> ANOM_METRICS
    LB --> APP
    ANOM_METRICS --> APP
    LB --> API
```

`app` y `api` **no calculan nada**: solo leen los `.parquet` de
`reports/results/` (subconjunto demo versionado) o `data/artifacts/live/`
(escrito por `scripts/refresh_data.py`). Eso es lo que permite desplegar la
app en Streamlit Community Cloud sin arrastrar `torch`, `prophet` ni
`statsforecast` — el árbol de dependencias por capa está en
`pyproject.toml` (extras `ml`, `deep`, `app`, `api`).

### El protocolo de backtesting, y por qué está diseñado así

Un backtest de forecasting se rompe casi siempre por un motivo: algo que el
modelo no debería conocer en el momento de predecir se cuela igualmente,
normalmente en el escalado, la imputación o la selección de features. La
respuesta del proyecto no es "tener cuidado" — es hacer que la fuga sea
**estructuralmente imposible**, con tres mecanismos, de más a menos fuertes:

1. **Ausencia física.** El dato prohibido no existe en la estructura que
   recibe el consumidor. `FutrFrame` —lo único que ve un modelo del futuro—
   solo contiene las columnas declaradas `futr_exog`; la objetivo y las
   `hist_exog` no están "vacías por convenio", están ausentes del tipo.
2. **Constructor único.** `RollingOriginSplitter` es el único camino para
   obtener una partición temporal: no existe ninguna función que acepte
   máscaras booleanas ni fechas sueltas, así que no hay forma de escribir un
   split aleatorio por accidente.
3. **Aserción en el único camino de código.** El motor comprueba, en cada
   ventana: que el `train` recortado no cruza el `cutoff`
   (`_assert_train_at_cutoff`), que las exógenas futuras entregadas no traen
   la objetivo ni una `hist_exog` (`_assert_futr_frame`), que ninguna
   predicción cae en el pasado ya conocido (`_assert_prediction_after_cutoff`)
   y que un ajuste reutilizado por la política de refit no es posterior a su
   ventana (`_assert_fitted_at_or_before_cutoff`). Una violación no se
   registra como "modelo fallido": el run entero se detiene con
   `LeakageError`, porque un run con fuga no produce resultados publicables.

El denominador de MASE sigue la misma disciplina: se calcula **por serie y
por ventana**, con el tramo de entrenamiento exacto de esa ventana
(`seasonal_naive_mae` sobre `[train_start, cutoff]`), nunca sobre el test ni
sobre la serie completa — las dos variantes más comunes de "MASE con fuga" en
repositorios de forecasting. Protocolo completo, con las cuatro garantías
detalladas y las cuatro fases de fuga posible, en
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

### Por qué la métrica de anomalías no es F1 puntual

El *point-adjusted F1* —marcar como detectado un segmento entero si el
detector acierta un solo punto suyo— es el estándar de facto en muchos
benchmarks publicados, y está roto: basta con emitir **ruido aleatorio** para
obtener un F1 comparable al estado del arte, porque salpicar puntos al azar
toca casi cualquier segmento largo. El proyecto no lo implementa, y lo prueba
con un test adversario (`tests/unit/evaluation/test_anomaly_metrics.py`) que
reproduce esa inflación: ruido aleatorio saca un F1-PA > 0.55, más de 5 veces
el F1 honesto sobre los mismos datos.

En su lugar, cinco familias de métrica que fallan de formas distintas —para
que un detector solo parezca bueno si lo es bajo criterios que no comparten
sesgo—: precisión/recall **por rangos** (Tatbul et al. 2018), **de
afiliación** (Huet et al. 2022, mide distancia y no solape), **VUS-PR**
(Paparrizos et al. 2022, integra sobre umbral y tolerancia temporal a la vez),
AUC-PR puntual como referencia mínima, y un bloque **operativo** (retardo de
detección, falsas alarmas por 1000 observaciones) que es lo que de verdad
decide si un detector se despliega. Definiciones formales y la comprobación
empírica de que el orden cambia según cuál se mire, en
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) y
[`docs/ANOMALY_FINDINGS.md`](docs/ANOMALY_FINDINGS.md).

### Árbol de módulos (resumen)

```
src/chronolab/
├── data/        ingesta (REE, Open-Meteo, UCI, Binance, sintético), alineado UTC/DST, calidad
├── panel.py     Panel / PanelSpec / FutrFrame — el contrato que envuelve todo DataFrame
├── features/    primitivas causales (lag/roll/expand/ewm), álgebra de disponibilidad temporal
├── models/      6 familias de forecaster tras un protocolo común (ver docs/MODEL_CARDS.md)
├── evaluation/  splitter, motor de backtest, métricas, Diebold-Mariano, tuning (Optuna)
├── anomaly/     inyección sintética, 4 detectores, umbralización conformal
├── artifacts/   esquemas y lectura/escritura de reports/results/
├── viz/         gráficos reutilizados por notebooks y app
├── app/         Streamlit — solo lee artefactos
└── api/         FastAPI de solo lectura sobre data/artifacts/live/
```

---

## Instalación y uso

```bash
uv sync                 # core + dev: suficiente para lint, typecheck y tests
uv sync --all-extras     # + modelos (ml, deep) y la app/API
```

| Comando | Qué hace |
|---|---|
| `make install` | Instala el entorno completo |
| `make lint` | Lint y comprobación de formato con `ruff` |
| `make typecheck` | `mypy --strict` sobre `src/` |
| `make test` | Suite completa, con cobertura |
| `make test-fast` | Excluye tests `slow` y `network` |
| `make app` | Levanta el dashboard Streamlit (`localhost:8501`) |
| `make clean` | Borra cachés |

Reproducir el leaderboard y los hallazgos de anomalías desde cero (requiere
los extras `ml`/`deep`):

```bash
uv run --extra ml python scripts/run_ml_feature_analysis.py   # tuning + baselines + ML
uv run --extra ml --extra deep python scripts/run_deep_analysis.py  # + NHITS/TFT/PatchTST/LSTM
uv run --extra ml --extra deep python scripts/run_anomaly_eval.py   # 4 detectores × 6 tipos
```

Docker (API de solo lectura + dashboard, sin los extras `ml`/`deep`):

```bash
docker compose up            # api en :8000, app en :8501
```

---

## Limitaciones y qué haría diferente

Esto se escribe para que se use bien, no para que se lea bien.

**Lo que el proyecto no cubre todavía:**

- **Ningún número está calculado sobre datos reales.** Todo lo de arriba
  viene de `tests.fixtures.synthetic.make_hourly_panel`. Los conectores a
  REE, Open-Meteo, UCI y Binance existen y están testeados, pero
  `scripts/refresh_data.py` es el único camino que los usa y su salida está
  gitignorada. El primer paso real para producción es repetir este mismo
  protocolo sobre demanda eléctrica real y ver cuánto de esto se sostiene.
- **`chronolab.data.assemble.build_panel` no está implementado.** El
  constructor que debería garantizar que un `Panel` tiene rejilla completa y
  sin duplicados es un stub. Hoy nada impide construir un `Panel` con un
  hueco no representado como fila `NaN`, y eso corrompe silenciosamente el
  denominador de MASE y cualquier feature de lag (verificado: quitar una
  sola fila de una serie perfectamente periódica cambia su denominador de
  `0.0` a `0.28`). Es la pieza que más urge antes de correr contra datos
  reales, donde los huecos son la norma y no la excepción.
- **El leaderboard compartido mezcla presciencia perfecta con predicción
  real**, sin distinguirlo en el artefacto (ver el † de la tabla de
  arriba). `RealizedFutrProvider` existe para dar una *cota superior* citable,
  no para colarse en la comparación por defecto — y lo hace, porque el aviso
  que lo impide se silencia a mano en los scripts que generan
  `leaderboard.parquet`. La corrección es correr con `SimulatedForecastProvider`
  (pronóstico con error realista) para la comparación principal, y reservar
  `RealizedFutrProvider` para una ablación etiquetada aparte.
- **El leaderboard descansa en 2 ventanas de holdout (144 observaciones).**
  El motor implementa el test de Diebold-Mariano con corrección HLN, pero no
  se aplica al leaderboard — así que un 1–3 % de diferencia de MASE entre los
  tres primeros puestos se presenta sin ningún contraste de significancia.
- **Dos detectores de anomalía (IsolationForest, LSTM-Autoencoder) no están
  calibrados al umbral nominal** — marcan la mitad del holdout a `α = 0.05`
  en vez del 5 % esperado (ver Hallazgos). Publicarlos junto a un detector sí
  calibrado (Conformal) sin corregirlo es comparar cosas distintas con la
  misma vara.
- **CI no ejercita el código de modelado.** El job de calidad instala solo
  `core + dev + api`; los extras `ml`/`deep` —y por tanto los seis
  adaptadores de modelo y los cuatro detectores— nunca corren en GitHub
  Actions. La suite completa pasa en local (1274 tests), pero eso depende
  del entorno de una máquina concreta, no de lo que CI certifica en verde.

**Qué haría distinto si empezara de nuevo:**

- Implementar `build_panel` con validación pandera desde el primer commit de
  `data/`, no como una promesa en un docstring — es la pieza sobre la que
  descansa todo lo demás y es barata de hacer pronto, cara de arreglar tarde.
- Fijar `n_windows`/`holdout_windows` mucho más alto desde el principio
  (mínimo 15–20 de holdout) y hacer del test de Diebold-Mariano parte del
  propio `build_leaderboard`, no un artefacto aparte que hay que acordarse de
  mirar.
- Medir la tasa de marcado de cada detector de anomalías sobre un tramo
  limpio **antes** de fijar el umbral de comparación, no después de ver los
  resultados — habría capturado el problema de calibración de IsolationForest
  y el autoencoder en la primera corrida, no en una auditoría posterior.

**Qué haría falta para producción, más allá de lo anterior:**

- `run_id` persistido de verdad (`artifacts/writer.py` es hoy un stub) y una
  tabla `runs` que permita rastrear cada número publicado a su configuración
  exacta.
- Monitorización de deriva de los modelos ajustados y una cadencia de
  reentrenamiento definida — nada de esto existe hoy; el "refresco" de
  `scripts/refresh_data.py` recalibra el detector conformal pero no
  reentrenar los forecasters.
- Un registro de modelos real (`conf/models.yaml` → `models/registry.py`,
  hoy también un stub) en vez de instanciar cada `Forecaster` a mano en cada
  script.

---

## Referencias

- Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
  forecast accuracy.* International Journal of Forecasting, 22(4), 679–688.
  — definición de MASE.
- Makridakis, S., et al. (2020). *The M4 Competition.* International Journal
  of Forecasting. — definición de sMAPE usada en el proyecto.
- Gneiting, T., & Raftery, A. E. (2007). *Strictly proper scoring rules,
  prediction, and estimation.* JASA. — pérdida pinball y CRPS como reglas de
  puntuación propias.
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy.*
  Journal of Business & Economic Statistics. Corrección de muestra finita:
  Harvey, D., Leybourne, S., & Newbold, P. (1997). *Testing the equality of
  prediction mean squared errors.*
- Tatbul, N., et al. (2018). *Precision and recall for time series.*
  NeurIPS. — métricas por rangos.
- Huet, A., Navarro, J. M., & Rossi, D. (2022). *Local evaluation of time
  series anomaly detection algorithms.* KDD. — precisión/recall de afiliación.
- Paparrizos, J., et al. (2022). *Volume under the surface: a new accuracy
  evaluation measure for time-series anomaly detection.* PVLDB / TSB-UAD. —
  VUS-PR.
- Kim, S., et al. (2022). *Towards a rigorous evaluation of time-series
  anomaly detection.* AAAI. — crítica del *point-adjusted F1*.
- Ansari, A. F., et al. (2024). *Chronos: Learning the language of time
  series.* — modelo fundacional zero-shot (`amazon/chronos-bolt-small`).
- Challu, C., et al. (2023). *NHITS: Neural hierarchical interpolation for
  time series forecasting.* AAAI.
- Lim, B., et al. (2021). *Temporal Fusion Transformers for interpretable
  multi-horizon time series forecasting.* IJoF.
- Nie, Y., et al. (2023). *A time series is worth 64 words: long-term
  forecasting with transformers* (PatchTST). ICLR.
- Ecosistema Nixtla: `statsforecast`, `mlforecast`, `neuralforecast` —
  adaptadores de modelo del proyecto, envueltos tras un protocolo propio
  (`chronolab.models.protocols.Forecaster`).

## Control de versiones

Los commits los gestiona una persona, no el asistente. Las reglas están en
[`CLAUDE.md`](CLAUDE.md) y se hacen cumplir desde `.claude/settings.json`.

## Licencia

MIT.
