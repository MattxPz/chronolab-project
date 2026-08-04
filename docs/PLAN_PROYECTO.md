# Plataforma de Forecasting y Detección de Anomalías en Series Temporales
## Plan maestro de ejecución

> Nombre sugerido del repo: **`chronolab`** (o `tsforge`, `pulsecast`). Corto, pronunciable, sin colisiones obvias en PyPI.

---

## 0. La decisión que define el proyecto

Antes de escribir una línea de código, hay que entender **por qué la mayoría de proyectos de forecasting en GitHub no impresionan a nadie**:

1. **No tienen baseline.** Muestran un LSTM con MAPE de 4% y nunca dicen que `SeasonalNaive` daba 3.8%.
2. **Tienen fuga de información (leakage).** Escalan con la media de todo el dataset, o hacen `train_test_split` aleatorio sobre datos temporales, o calculan features con ventanas centradas.
3. **Evalúan anomalías con métricas desacreditadas.** El "point-adjusted F1" está reconocido en la literatura (TSB-AD, NeurIPS 2024) como una métrica que infla resultados hasta hacer que ruido aleatorio parezca state-of-the-art.
4. **Usan precios de criptomonedas.** Una serie financiera de alta frecuencia es prácticamente un paseo aleatorio: el modelo naive gana casi siempre y quien revisa lo sabe.

Tu proyecto destaca si hace exactamente lo contrario. **El diferenciador no son los modelos, es el rigor del marco de evaluación.** Los modelos son commodities (tres líneas con Nixtla); un backtesting honesto con origen rodante, métricas escala-independientes y evaluación de anomalías por eventos no lo es.

### Elección de dominio

| Opción | Veredicto |
|---|---|
| **Demanda eléctrica horaria** | ✅ **Recomendada como serie principal.** Estacionalidad múltiple clara (diaria, semanal, anual), covariables externas obvias (temperatura, calendario, festivos), señal real predecible, datasets públicos abundantes. |
| Demanda retail (M5 / Favorita) | ✅ Buena segunda serie. Intermitente, jerárquica, otro régimen de dificultad. |
| Precios de cripto | ⚠️ Úsala **solo como serie de contraste**, y di explícitamente en el README: *"incluida deliberadamente para mostrar un caso donde ningún modelo bate al naive; saber reconocer esto es parte del trabajo"*. Eso te suma puntos en vez de restarte. |

### Fuentes de datos concretas

| Fuente | Acceso | Rol |
|---|---|---|
| **UCI ElectricityLoadDiagrams 2011-2014** | Descarga directa, 370 clientes, 15 min | Serie principal offline, reproducible siempre |
| **REE apidatos** (`apidatos.ree.es`) | API pública, sin token | Demanda eléctrica España, actualizada → habilita el "casi tiempo real" |
| **Open-Meteo** (`archive-api.open-meteo.com`) | API pública, sin token | Temperatura histórica y forecast → **covariable exógena futura conocida** |
| **ENTSO-E Transparency** | Token gratuito | Alternativa europea con más granularidad |
| **Binance klines** (`api.binance.com/api/v3/klines`) | Público, sin token | Serie de contraste cripto |

La combinación **demanda + temperatura + calendario** es la que hace el proyecto interesante: te permite mostrar modelos con regresores exógenos, que es donde ARIMA, LightGBM y TFT realmente se diferencian.

---

## 1. Stack tecnológico recomendado (estado del arte 2026)

### La decisión arquitectónica clave: adoptar el contrato de datos de Nixtla

`statsforecast`, `mlforecast` y `neuralforecast` comparten el mismo formato largo (`unique_id`, `ds`, `y`) y la misma API `.fit()/.predict()`. Si adoptas ese formato como **contrato interno de datos**, obtienes:

- Modelos clásicos, ML y deep learning intercambiables sin código de pegamento.
- Backtesting (`cross_validation`) unificado y ya probado.
- Intervalos conformales nativos (`ConformalIntervals`) — la base de tu detector de anomalías.
- Poder envolver Prophet, tu LSTM propio y un modelo fundacional bajo el mismo protocolo `Forecaster`.

Esto reduce el proyecto de ~6000 líneas frágiles a ~2000 líneas sólidas.

### Tabla de stack

| Capa | Elección | Por qué |
|---|---|---|
| **Runtime** | Python 3.12 | Compatibilidad amplia con el ecosistema TS |
| **Gestor de entorno** | **`uv`** | Estándar de facto en 2026; `uv sync` reproducible, 10-100× más rápido que pip |
| **Datos** | pandas + pyarrow (parquet) | Parquet para caché; polars opcional si quieres presumir |
| **Validación de esquema** | `pandera` | Contratos de datos verificables → demuestra madurez de ingeniería |
| **Baselines** | `statsforecast`: `Naive`, `SeasonalNaive`, `WindowAverage`, `HistoricAverage` | Innegociable. Es la referencia contra la que se mide todo |
| **Estadísticos** | `statsforecast`: `AutoARIMA`, `AutoETS`, `AutoTheta`, **`MSTL`** | MSTL maneja estacionalidad múltiple (24h + 168h), que es exactamente tu caso |
| **Prophet** | `prophet` | Sigue mantenido (release ene-2026). Inclúyelo porque es el estándar de industria y porque a menudo **pierde** — eso también es un hallazgo |
| **ML** | **`mlforecast`** + LightGBM y XGBoost | Genera lags/rollings/date-features **sin leakage** y maneja forecasting recursivo vs directo correctamente. Hacerlo a mano es donde el 80% de los proyectos se rompen |
| **Deep Learning** | **`neuralforecast`**: `NHITS`, `TFT`, `PatchTST` + un **LSTM propio en PyTorch** | NHITS/PatchTST son SOTA-por-esfuerzo; el LSTM artesanal demuestra que sabes lo que hay debajo, no solo llamar APIs |
| **Modelo fundacional** (diferenciador) | **Chronos-2** o **Chronos-Bolt** vía Hugging Face; alternativa TimesFM-2.5 | Zero-shot, sin entrenamiento. En 2026 es *la* pregunta que hace un entrevistador: "¿lo comparaste contra un foundation model?" Chronos-Bolt corre en CPU |
| **Anomalías (principal)** | Residuos + **intervalos conformales** | Rigor estadístico: tasa de falsos positivos controlada por construcción |
| **Anomalías (comparativas)** | `PyOD` (IsolationForest, LSTM-AE, MatrixProfile) + `stumpy` | PyOD 3.x incorpora detectores de series temporales rankeados por el benchmark TSB-AD |
| **Métricas de anomalía** | **VUS-PR** y F1 por rangos/afiliación | Evita el point-adjusted F1 desacreditado. Este detalle solo lo pone quien ha leído la literatura |
| **Tracking** | MLflow (local, `mlruns/` gitignored) | Vistoso y estándar; alternativa ligera: resultados en parquet + tabla markdown |
| **Dashboard** | **Streamlit** | Mejor que Gradio para multi-panel con estado, caché (`@st.cache_data`) y filtros. Gradio es superior para demo de *un* modelo; tú tienes ocho |
| **Gráficos** | Plotly | Zoom e inspección interactiva sobre series largas; imprescindible para señalar anomalías |
| **API opcional** | FastAPI | Endpoints `/forecast` y `/anomalies` |
| **Actualización periódica** | GitHub Actions con `schedule` (cron) | El truco elegante: el repo se actualiza solo, la demo nunca está muerta |
| **Calidad** | `ruff` (lint+format), `mypy`, `pytest`, `pre-commit` | |
| **CI** | GitHub Actions | Lint + tests + smoke test del pipeline |
| **Contenedor** | Docker + `docker-compose` | Reproducibilidad demostrable |
| **Deploy** | Streamlit Community Cloud (gratis) o Hugging Face Spaces | Enlace vivo en el README = 10× más impacto |

---

## 2. Arquitectura del sistema

```
                    ┌──────────────────────────────────────┐
                    │           CAPA DE DATOS              │
                    │  DataSource (Protocol)               │
                    │  ├─ UCIElectricitySource             │
                    │  ├─ REEDemandSource      (API)       │
                    │  ├─ OpenMeteoSource      (exógena)   │
                    │  └─ BinanceSource        (contraste) │
                    │  → validación pandera → parquet caché │
                    └───────────────┬──────────────────────┘
                                    │  formato largo (unique_id, ds, y, exog...)
                    ┌───────────────▼──────────────────────┐
                    │        MOTOR DE BACKTESTING          │
                    │  Origen rodante (expanding/sliding)  │
                    │  + gap anti-leakage + horizonte h     │
                    │  → predicciones fuera de muestra      │
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐        ┌─────────▼─────────┐       ┌────────▼─────────┐
│  FORECASTERS   │        │     MÉTRICAS      │       │   ANOMALÍAS      │
│ Baselines      │        │ MASE, RMSE, MAE   │       │ Conformal        │
│ AutoARIMA/ETS  │        │ sMAPE, MAPE(*)    │       │ IsolationForest  │
│ MSTL, Prophet  │        │ Pinball / CRPS    │       │ LSTM-AE          │
│ LGBM, XGBoost  │        │ Diebold-Mariano   │       │ MatrixProfile    │
│ NHITS,TFT,LSTM │        │                   │       │ → VUS-PR, F1-rango│
│ Chronos (0shot)│        └───────────────────┘       └──────────────────┘
└────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  ARTEFACTOS (parquet + MLflow)       │
                    └───────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  STREAMLIT · FastAPI · GH Actions    │
                    └──────────────────────────────────────┘
```

### Estructura del repositorio

```
chronolab/
├── README.md                    ← la pieza más importante del repo
├── LICENSE                      (MIT)
├── pyproject.toml
├── uv.lock
├── Makefile
├── .pre-commit-config.yaml
├── .gitignore
├── CLAUDE.md                    ← reglas para Claude Code (incl. "no hagas git")
├── .claude/
│   ├── settings.json            ← permisos: deny git commit/push
│   └── commands/                ← slash commands propios
├── .github/workflows/
│   ├── ci.yml
│   └── refresh-data.yml         ← cron: actualiza datos y re-puntúa
├── conf/
│   ├── datasets.yaml
│   ├── models.yaml
│   └── backtest.yaml
├── data/                        (gitignored salvo .gitkeep)
│   ├── raw/  interim/  processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_anomaly_analysis.ipynb
├── src/chronolab/
│   ├── config.py
│   ├── data/          sources.py  loaders.py  schemas.py  calendar.py
│   ├── features/      builders.py
│   ├── models/        base.py  baselines.py  statistical.py  ml.py
│   │                  deep.py  foundation.py  registry.py
│   ├── anomaly/       base.py  conformal.py  isolation.py
│   │                  autoencoder.py  matrix_profile.py
│   ├── evaluation/    backtest.py  metrics.py  anomaly_metrics.py  stats_tests.py
│   ├── viz/           plots.py
│   ├── app/           main.py  pages/  components/
│   └── api/           service.py
├── tests/
├── reports/
│   ├── figures/
│   └── results/       leaderboard.parquet
└── scripts/           run_backtest.py  run_anomaly.py  refresh_data.py
```

---

## 3. Plan por fases

Estimaciones para una persona con Claude Code asistiendo. La columna "riesgo" indica dónde se hunden los proyectos.

### FASE 0 — Encuadre y decisiones · ~0.5 día · riesgo: bajo
- Elegir dominio y 2-3 series concretas.
- Definir **horizonte** (`h=24` u `h=168` para demanda horaria) y **frecuencia**.
- Definir criterios de éxito medibles: *"al menos un modelo bate a SeasonalNaive en MASE con significancia estadística (Diebold-Mariano p<0.05) en las 3 series"*.
- Escribir el borrador del README **antes** del código (README-driven development). Te obliga a saber qué estás construyendo.

**Entregable:** `README.md` con alcance, y `conf/datasets.yaml`.

### FASE 1 — Andamiaje e higiene · ~1 día · riesgo: bajo
- `uv init`, `pyproject.toml`, layout `src/`.
- ruff + mypy + pytest + pre-commit + Makefile.
- CI mínimo en GitHub Actions.
- `CLAUDE.md` y `.claude/settings.json` (ver §5 — control de versiones manual).

**Entregable:** repo que pasa `make lint test` en verde y en vacío.

### FASE 2 — Capa de datos · ~1.5 días · riesgo: medio
- Protocolo `DataSource` con `fetch() -> pd.DataFrame` en formato largo.
- Implementar UCI (descarga + parseo), REE (API), Open-Meteo (exógenas), Binance.
- Caché en parquet con invalidación por fecha.
- Validación con `pandera`: tipos, rango, monotonía del índice, huecos.
- Manejo explícito de: huecos temporales, cambios de horario (DST — trampa clásica en series horarias europeas), duplicados, ceros vs NaN.

**Riesgo real:** el cambio de hora crea horas duplicadas y horas faltantes dos veces al año. Si no lo manejas, todos los modelos estacionales se desalinean. Trabaja internamente en UTC y convierte solo para mostrar.

**Entregable:** `chronolab.data` + tests con datos sintéticos.

### FASE 3 — EDA y descomposición · ~1 día · riesgo: bajo
- Descomposición STL / MSTL (múltiples estacionalidades).
- ACF/PACF, periodogramas, correlación con temperatura (relación en "U" con la demanda: sube con frío y con calor).
- Detección visual de huecos, festivos, quiebres estructurales (COVID si el rango lo cubre).
- Estadísticos de dificultad de la serie: entropía espectral, fuerza de tendencia y estacionalidad.

**Entregable:** `notebooks/01_eda.ipynb` + figuras en `reports/figures/`.

### FASE 4 — ⭐ Motor de backtesting y métricas · ~2 días · riesgo: ALTO
Esta es la fase que determina si el proyecto vale algo.

- **Validación de origen rodante** (rolling origin): ventana expansiva y deslizante, `n_windows`, `step_size`, `h`, y un `gap` opcional.
- Garantías anti-leakage:
  - Escaladores ajustados **solo** con train de cada ventana.
  - Features de lag y rolling calculadas solo hacia atrás.
  - Variables exógenas separadas en *conocidas a futuro* (calendario, forecast de temperatura) vs *solo históricas*.
- Métricas: **MASE** (principal — escala-independiente y comparable entre series), RMSE, MAE, sMAPE. MAPE solo con advertencia (explota con valores cercanos a cero).
- Métricas probabilísticas: pinball loss por cuantil, cobertura empírica de los intervalos, CRPS aproximado.
- Test de **Diebold-Mariano** para decir si la diferencia entre dos modelos es real o ruido.
- Todo se persiste en `reports/results/leaderboard.parquet` con semilla, tiempo de entrenamiento y versión de config.

**Entregable:** `chronolab.evaluation` con tests unitarios que verifican explícitamente que no hay leakage.

### FASE 5 — Baselines y modelos estadísticos · ~1 día · riesgo: bajo
- `Naive`, `SeasonalNaive`, `WindowAverage`, `HistoricAverage`.
- `AutoARIMA`, `AutoETS`, `AutoTheta`, `MSTL` (con `AutoARIMA` como componente de tendencia).
- Prophet con festivos del país y regresor de temperatura.
- **Primera tabla del leaderboard.** Muchas veces MSTL o SeasonalNaive ya son difíciles de batir: documéntalo.

### FASE 6 — Modelos de ML · ~2 días · riesgo: medio
- `mlforecast` con LightGBM y XGBoost.
- Features: lags (1, 24, 168), medias/desviaciones móviles, expanding, features de calendario (hora, día de semana, mes, festivo, sin/cos de Fourier), temperatura y grados-día.
- Comparar estrategia **recursiva vs directa** (un modelo por horizonte) — un análisis que casi nadie hace y que se nota.
- Importancia de features (nativa + SHAP) para el storytelling.
- Tuning con Optuna sobre las ventanas de validación, nunca sobre el test final.

### FASE 7 — Deep learning y modelo fundacional · ~2.5 días · riesgo: medio-alto
- `neuralforecast`: `NHITS`, `TFT` (interpretable, con atención por variable), `PatchTST`.
- **LSTM propio en PyTorch**: dataset con ventanas, entrenamiento con early stopping, escalado por serie. No hace falta que gane; hace falta que esté bien hecho y que lo compares honestamente.
- **Chronos-2 / Chronos-Bolt zero-shot**: sin entrenamiento, contexto → predicción. Añade una columna al leaderboard titulada *"zero-shot"* y comenta el resultado. Si un modelo sin entrenar queda cerca de tu TFT ajustado, **dilo**: es el hallazgo más interesante que puedes reportar.
- Registrar coste computacional (segundos de entrenamiento, parámetros) junto a la precisión: el eje precisión/coste es el que importa en producción.

### FASE 8 — ⭐ Detección de anomalías · ~2.5 días · riesgo: ALTO
Enfoque en capas, de más riguroso a más exploratorio:

1. **Residuos + intervalos conformales** (principal). Predices con el mejor modelo, construyes intervalos conformales sobre residuos de calibración, y marcas como anómalo lo que cae fuera del intervalo (1-α). Ventaja: la tasa de falsos positivos está controlada teóricamente, sin asumir normalidad.
2. **Isolation Forest** sobre un vector de features de ventana (valor, residuo, z-score móvil, derivada, energía espectral local).
3. **Autoencoder / LSTM-AE** sobre ventanas: error de reconstrucción como score.
4. **Matrix Profile** (`stumpy`): discords, robusto y sin entrenamiento.

**Evaluación (aquí está el diferenciador):**
- Inyectar anomalías sintéticas de tipos distintos (pico puntual, cambio de nivel, cambio de varianza, desfase estacional, congelación del sensor) para tener ground truth.
- Evaluar con **VUS-PR** y F1 por rangos/afiliación. **No** usar point-adjusted F1; y menciona en el README por qué lo evitas, citando la literatura de benchmarking.
- Curva precisión-recall por tipo de anomalía: qué detector captura qué. Ese gráfico es oro para una entrevista.

### FASE 9 — Dashboard Streamlit · ~2.5 días · riesgo: medio
Páginas:
1. **Overview** — serie, descomposición, estadísticos, calidad de datos.
2. **Forecast** — selector de serie/horizonte/modelos, predicciones superpuestas con bandas de incertidumbre, zoom.
3. **Leaderboard** — tabla ordenable de métricas, gráfico precisión vs tiempo de cómputo, resultados del test de Diebold-Mariano.
4. **Anomalías** — serie con puntos marcados, slider de sensibilidad (α), comparativa entre detectores, tabla de eventos con severidad.
5. **Explicabilidad** — importancia de features, pesos de atención del TFT, descomposición de la predicción.

Claves técnicas: `@st.cache_data` sobre la carga de artefactos, **nunca entrenar dentro de la app** (carga resultados precomputados), estado en `st.session_state`, y un modo "demo" con datos incluidos para que funcione sin red.

### FASE 10 — Casi tiempo real y despliegue · ~1.5 días · riesgo: medio
- Workflow `refresh-data.yml` con cron (p. ej. cada 6 h): descarga datos nuevos, re-puntúa con los modelos guardados, actualiza artefactos.
- FastAPI opcional: `POST /forecast`, `POST /anomalies`, `GET /health`.
- Dockerfile multi-stage + `docker-compose.yml`.
- Deploy en Streamlit Community Cloud o HF Spaces. **Enlace vivo en la cabecera del README.**

### FASE 11 — Documentación y presentación · ~1.5 días · riesgo: bajo (impacto: ALTO)
- README con: GIF de la demo, enlace en vivo, la **tabla de resultados en la primera pantalla**, decisiones de diseño, y una sección de *"limitaciones y qué haría diferente"* (esto proyecta seniority más que cualquier modelo).
- `docs/METHODOLOGY.md`: protocolo de backtesting, definición exacta de cada métrica.
- Model cards por modelo.
- Un post o notebook narrativo con los 3-4 hallazgos principales.

### FASE 12 — Extras diferenciadores (opcionales, por orden de retorno)
1. **Reconciliación jerárquica** (`hierarchicalforecast`) si tienes series agregables.
2. **Ensemble / model averaging** con pesos por ventana.
3. **Detección de drift** y disparador de reentrenamiento.
4. **Forecasting probabilístico completo** con evaluación por cuantiles y diagramas de fiabilidad.
5. **Reporte automático en PDF** por serie.

---

## 4. Cronograma sugerido

| Semana | Contenido | Estado del repo |
|---|---|---|
| 1 | Fases 0-3 | Datos limpios + EDA publicable |
| 2 | Fases 4-5 | **Primer leaderboard real** — ya es un repo defendible |
| 3 | Fases 6-7 | Comparativa completa de modelos |
| 4 | Fase 8 | Anomalías con evaluación rigurosa |
| 5 | Fases 9-10 | Demo en vivo |
| 6 | Fases 11-12 | Pulido y extras |

Si el tiempo aprieta: **fases 0-5 + 8 + 9 ya son un proyecto excelente.** El deep learning es lo primero que se recorta, no el backtesting.

---

## 5. Control de versiones: los commits los haces tú

Como vas a gestionar el repositorio manualmente, conviene blindar a Claude Code para que no toque git. Dos capas:

**`CLAUDE.md`** (raíz del repo):
```markdown
## Reglas de control de versiones — INNEGOCIABLES
- NUNCA ejecutes `git add`, `git commit`, `git push`, `git merge`, `git rebase`,
  `git checkout -b`, `git reset` ni ningún comando `gh`.
- El humano gestiona todo el control de versiones y el repositorio remoto.
- Al terminar cada tarea, imprime:
  1. Lista de archivos creados/modificados/eliminados.
  2. Un mensaje de commit sugerido en formato Conventional Commits.
- Sí puedes usar comandos de solo lectura: `git status`, `git diff`, `git log`.
```

**`.claude/settings.json`**:
```json
{
  "permissions": {
    "deny": [
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "Bash(git reset:*)",
      "Bash(git rebase:*)",
      "Bash(gh:*)"
    ],
    "allow": [
      "Bash(git status)",
      "Bash(git diff:*)",
      "Bash(git log:*)",
      "Bash(uv run:*)",
      "Bash(make:*)"
    ]
  }
}
```

La capa de permisos es la que realmente bloquea; el `CLAUDE.md` explica el porqué y define el comportamiento sustituto (sugerir el mensaje de commit).

---

## 6. Criterios de calidad del entregable final

- [ ] `uv sync && make test` funciona en una máquina limpia.
- [ ] Existe un baseline y **está en la tabla principal**, no escondido.
- [ ] El backtesting es de origen rodante, no un único split.
- [ ] Hay al menos un test que verificaría el leakage si lo introdujeras.
- [ ] Las métricas incluyen MASE, no solo MAPE.
- [ ] La evaluación de anomalías no usa point-adjustment (y el README dice por qué).
- [ ] La app arranca sin conexión a internet gracias a datos de demo incluidos.
- [ ] El README enseña resultados antes de enseñar instrucciones de instalación.
- [ ] Hay una sección de limitaciones escrita con honestidad.
- [ ] Hay un enlace a la demo desplegada.