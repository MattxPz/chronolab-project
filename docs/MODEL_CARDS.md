# Model cards

Por cada modelo de forecasting y cada detector de anomalías: qué supuestos
hace, qué le cuesta y en qué situación conviene o no usarlo. Los números de
coste y precisión son del backtest de referencia descrito en
[`README.md`](../README.md#resultados-backtest-de-referencia) — panel
sintético de 3 series, 6 ventanas, `h=24` — y solo son válidos como
comparación relativa *dentro de ese run*, no como cifra absoluta transferible
a otro dataset. Los cuatro modelos marcados † se ajustaron con temperatura
futura **real** (`RealizedFutrProvider`), no un pronóstico degradado — ver
[Limitaciones](../README.md#limitaciones-y-qué-haría-diferente).

Todos implementan el mismo protocolo (`chronolab.models.protocols.Forecaster`
/ `FittedForecaster`): `fit(train, h)` recibe únicamente el panel recortado a
`ds ≤ cutoff`, `predict(futr, quantiles)` devuelve `y_hat` y las columnas de
cuantil pedidas (`NaN` para las que el modelo no produce — nunca un intervalo
inventado).

---

## Baselines (`chronolab.models.baselines`)

Cinco modelos en numpy puro, sin dependencia de ninguna librería de
forecasting — el punto de referencia que el arnés necesita poder calcular a
mano, independiente de que `statsforecast` cambie de convención de un
release a otro. Los cinco: huecos internos rellenados hacia atrás (`ffill`,
nunca hacia adelante), intervalos gaussianos con la desviación creciendo con
el horizonte (Hyndman & Athanasopoulos, cap. 5.2), `n_params = None` (no hay
nada que ajustar por optimización).

### `NaiveForecaster`

- **Supuesto:** el valor de mañana es el de hoy. Ningún supuesto sobre
  estacionalidad o tendencia.
- **Coste:** `< 0.002 s` de ajuste y predicción — el suelo de coste del
  proyecto.
- **Resultado de referencia:** MASE 2.445 — el peor del leaderboard junto a
  `drift`, `window_average` y `historic_average`.
- **Cuándo usarlo:** como piso de comparación, nunca como modelo de
  producción salvo en series sin ninguna estructura (ruido puro, paseo
  aleatorio genuino).
- **Cuándo no:** cualquier serie con estacionalidad — que es la mayoría de
  demanda energética, tráfico web o similar.

### `SeasonalNaiveForecaster`

- **Supuesto:** el valor de hoy es el de hace `season` pasos. Es también el
  **denominador de MASE** (`seasonal_naive_mae`) — todo el leaderboard se
  mide relativo a este modelo.
- **Coste:** `< 0.002 s`.
- **Resultado de referencia:** con `season=168` (semanal), MASE 0.426 — noveno
  puesto de 19, por delante de `auto_arima`, `auto_ets`, `auto_theta` y `tft`.
  Con `season=24` (diario), MASE 0.727.
- **Cuándo usarlo:** siempre como referencia obligatoria — si un modelo más
  caro no lo bate, no se despliega. Es también un modelo de producción
  razonable cuando la estacionalidad domina y no hay presupuesto de cómputo.
- **Cuándo no:** series con tendencia fuerte o con estacionalidad que
  cambia de fase (el detector conformal, en `docs/ANOMALY_FINDINGS.md`,
  muestra que un residuo estacional-naive genera falsos positivos
  **garantizados** ante un desfase estacional real — no es solo un problema
  de precisión, es un problema de diseño experimental si se usa como modelo
  base de un detector).

### `WindowAverageForecaster`, `HistoricAverageForecaster`, `DriftForecaster`

- **Supuesto:** media móvil, media de toda la historia, o extrapolación
  lineal del último tramo — ninguno modela estacionalidad.
- **Coste:** `< 0.002 s` los tres.
- **Resultado de referencia:** MASE 2.40–2.45, indistinguibles entre sí y
  del `naive` puro sobre un panel con estacionalidad fuerte.
- **Cuándo usarlos:** diagnóstico de forma de la serie (si `drift` bate a
  `naive`, hay tendencia explotable; si no, no la hay), nunca como modelo de
  producción sobre una serie estacional.

---

## Estadísticos clásicos (`chronolab.models.adapters.statsforecast`)

Los cuatro son **univariados** — ninguno recibe exógenas; la temperatura la
usa `ProphetForecaster`, no estos. Todos admiten intervalos conformales
nativos (`ConformalIntervals`), pero **el intervalo fija el horizonte al
ajustar**: reutilizar el ajuste en una ventana con horizonte distinto desde
su propio cutoff falla explícitamente si `use_intervals=True`. Por eso, con
`refit_cost="expensive"` (política por defecto: un único ajuste por run), el
leaderboard de referencia corre estos cuatro con `use_intervals=False` — y
por eso no tienen métricas probabilísticas en la tabla (`n_obs_prob = 0`).

### `MSTLForecaster`

- **Supuesto:** descomposición STL con **múltiples** periodos estacionales
  simultáneos (aquí, 24 y 168), más un modelo (por defecto ARIMA) sobre la
  tendencia deseasonalizada. Es el único de los cuatro que modela
  explícitamente la doble estacionalidad diaria+semanal.
- **Coste:** 3.44 s de ajuste medio.
- **Resultado de referencia:** MASE 0.288 — tercer puesto global, el mejor de
  los modelos que **no** recibieron temperatura futura real.
- **Cuándo usarlo:** demanda energética, tráfico o cualquier serie con dos
  estacionalidades superpuestas y sin exógenas fuertes disponibles. Es el
  modelo base elegido para el módulo de anomalías precisamente por esto — ver
  `docs/ANOMALY_DESIGN.md`.
- **Cuándo no:** series con una sola estacionalidad clara (usar `AutoETS` o
  `AutoTheta`, más baratos) o donde una exógena (temperatura, precio) explica
  más varianza que el calendario.

### `AutoARIMAForecaster`

- **Supuesto:** proceso ARIMA estacional, orden seleccionado automáticamente.
  `approximation=True` por defecto (mínimos cuadrados condicionales en la
  búsqueda de órdenes en vez de máxima verosimilitud exacta) — 40–60× más
  rápido, a costa de una búsqueda algo menos exhaustiva.
- **Coste:** 5.18 s de ajuste medio en este panel (una sola serie horaria de
  720 puntos tarda ~0.5–0.7 s con `approximation=True`, ~32 s sin ella).
- **Resultado de referencia:** MASE 0.766 — el peor de los cuatro
  estadísticos clásicos, peor incluso que `seasonal_naive_168`.
- **Cuándo usarlo:** series cortas sin estacionalidad múltiple, donde el
  orden AR/MA no es obvio a priori.
- **Cuándo no:** aquí. Con dos estacionalidades y `168` pasos por ciclo
  semanal, el espacio de búsqueda de ARIMA estacional no está bien
  planteado — es el resultado observado, no una intuición.

### `AutoETSForecaster`

- **Supuesto:** suavizado exponencial con selección automática de
  error/tendencia/estacionalidad (aditivo o multiplicativo).
- **Coste:** 2.88 s de ajuste medio — el más barato de los cuatro
  estadísticos.
- **Resultado de referencia:** MASE 0.653.
- **Cuándo usarlo:** series con una sola estacionalidad y presupuesto de
  cómputo ajustado; suele ser un buen punto de partida barato antes de
  probar algo más caro.
- **Cuándo no:** doble estacionalidad (no la modela nativamente, a
  diferencia de MSTL).

### `AutoThetaForecaster`

- **Supuesto:** método Theta, descomposición en dos líneas theta
  (tendencia amplificada y suavizado exponencial simple).
- **Coste:** 13.96 s de ajuste medio.
- **Resultado de referencia:** MASE 0.507 — el mejor de los tres modelos
  estadísticos que no son MSTL.
- **Cuándo usarlo:** referencia rápida de "cuánto se puede sacar sin
  modelar la estructura explícitamente" — ganó competiciones M3/M4 en su
  categoría.
- **Cuándo no:** cuando ya se sabe que hay doble estacionalidad explotable
  (MSTL la aprovecha mejor, por 0.22 de MASE menos, a coste similar).

---

## Prophet (`chronolab.models.adapters.prophet`) †

- **Supuesto:** modelo aditivo (tendencia + estacionalidades de Fourier +
  festivos + regresores lineales), pensado para intervención humana y
  robustez a huecos, no para máxima precisión pura. Por defecto usa
  `temp_c` como regresor lineal (`regressors=("temp_c",)`) y festivos de
  España (`country_holidays="ES"`).
- **Coste:** 1.64 s de ajuste medio — el más barato de los modelos con
  intervalos probabilísticos activos.
- **Resultado de referencia:** MASE 0.279 — primer puesto. **Recibió
  temperatura futura real**, no un pronóstico: su ventaja sobre MSTL (0.288,
  sin exógenas) no es directamente atribuible al modelo sin esa salvedad.
- **Cuándo usarlo:** cuando hay una exógena razonablemente predecible con
  antelación suficiente (temperatura con pronóstico meteorológico a 24–48 h,
  no con el valor realizado) y se necesita que un no-especialista pueda
  ajustar festivos o puntos de cambio de tendencia a mano.
- **Cuándo no:** sin acceso a un pronóstico real de la exógena declarada —
  correrlo con `RealizedFutrProvider` y publicarlo sin la etiqueta de cota
  superior es el error que este mismo repositorio comete en su leaderboard
  de referencia (ver Limitaciones en el README).

---

## Gradient boosting vía mlforecast (`chronolab.models.adapters.mlforecast`)

`LightGBMForecaster` y `XGBoostForecaster` comparten adaptador y dos
decisiones de diseño: (1) lags/ventanas/diferencias de la propia objetivo se
delegan enteros en `mlforecast`, sin reimplementar la generación ni la
recursión a mano; (2) **nunca leen el `FutrFrame`** — reconstruyen calendario
y térmicas extendiendo la historia de entrenamiento, así que no hay forma de
que reciban presciencia sobre una exógena, a diferencia de Prophet/NHITS/
TFT/LSTM. Cada uno en dos estrategias:

- **`recursive`** — un único regresor, realimentado con sus propias
  predicciones en los lags cortos (`lag(y, 1)`).
- **`direct`** — `h` regresores independientes, uno por paso del horizonte,
  sin recursión (`max_horizon=h` de mlforecast).

### `LightGBMForecaster`

| Variante | MASE | Ajuste medio |
|---|--:|--:|
| `recursive` | 0.353 | 4.08 s |
| `direct` | 0.352 | **46.65 s** |

- **Supuesto:** la relación entre lags/ventanas y el valor futuro es no
  lineal y se puede aprender por árboles; no asume estacionalidad
  paramétrica, la aprende de los lags declarados (`LAGS = (1,2,3,24,48,168,336)`).
- **Coste:** la variante `direct` cuesta **11× más** que la `recursive` para
  un MASE prácticamente idéntico (0.352 vs 0.353) en este panel — porque
  `direct` ajusta un modelo por cada uno de los 24 pasos del horizonte.
- **Cuándo usarlo `recursive`:** casi siempre que se considere gradient
  boosting — el coste extra de `direct` no se traduce en precisión aquí.
- **Cuándo usarlo `direct`:** solo si hay evidencia de que el error se
  acumula de forma importante en `recursive` a horizontes largos (ver
  `docs/FEATURE_ANALYSIS.md`, sección de degradación por paso) y el
  presupuesto de cómputo lo permite.
- **Cuándo no usar ninguna:** paneles pequeños (pocas series, poca historia)
  donde MSTL o Prophet, más baratos de ajustar, ya bastan — aquí ambas
  variantes de LightGBM quedan por detrás de MSTL y Prophet en MASE *y* en
  coste.

### `XGBoostForecaster`

| Variante | MASE | Ajuste medio |
|---|--:|--:|
| `recursive` | 0.385 | 2.42 s |
| `direct` | 0.389 | 23.40 s |

- Mismos supuestos y misma estructura que LightGBM; en este panel queda
  sistemáticamente algo peor que LightGBM en las dos estrategias, con la
  variante `direct` de nuevo pagando ~10× el coste de `recursive` sin mejora.
- **Cuándo usarlo sobre LightGBM:** sin evidencia adicional en este dataset;
  la elección entre ambos suele decidirse por infraestructura existente
  (GPU, licencia, familiaridad del equipo) más que por precisión.

---

## Deep learning (`chronolab.models.adapters.neuralforecast`, `torch_lstm`)

Los tres de `neuralforecast` y el LSTM propio comparten una restricción dura:
predicen **exactamente** los `h` pasos siguientes a su cutoff de ajuste — no
hay forma de "estirar" la salida. Reutilizar el ajuste en una ventana
posterior desalinearía la predicción, y los cuatro lo detectan y fallan en
vez de alinear en silencio. Consecuencia: `refit_cost="expensive"` con
`refit_every=1` obligatorio, es decir, se paga el coste de ajuste completo en
**cada ventana** del backtest — no hay forma barata de correrlos.

### `NHITSForecaster` †

- **Supuesto:** interpolación jerárquica multi-escala; sin mecanismo de
  atención, más barato que un transformer para el mismo tamaño de contexto.
  `use_futr_exog=True` por defecto — recibió temperatura futura real en el
  run de referencia.
- **Coste:** 3.27 s de ajuste medio, 301 749 parámetros.
- **Resultado de referencia:** MASE 0.284 — segundo puesto, con la misma
  salvedad de presciencia perfecta que Prophet.
- **Cuándo usarlo:** panel con varias series, presupuesto de GPU/CPU
  disponible para reajustar en cada ventana, y una exógena futura con
  pronóstico real (no realizado).
- **Cuándo no:** panel pequeño (3 series aquí) — el coste de mantener
  305 000 parámetros para 3 series es difícil de justificar frente a MSTL,
  que no recibió exógenas y quedó a 0.004 de MASE.

### `TFTForecaster` †

- **Supuesto:** atención temporal + selección de variables interpretable
  (pesos de importancia por variable, atención por instante — únicos entre
  los modelos del proyecto en ofrecer esto nativamente).
- **Coste:** 33.80 s de ajuste medio, 56 468 parámetros — el modelo más caro
  del leaderboard después del LSTM propio.
- **Resultado de referencia:** MASE 0.622 — duodécimo puesto, peor que
  `seasonal_naive_168` (0.426) **pese a recibir temperatura futura real**.
- **Cuándo usarlo:** cuando la interpretabilidad nativa (qué variable pesó,
  cuándo se prestó atención) es un requisito del producto, no solo la
  precisión — es la única razón que justifica su coste aquí.
- **Cuándo no:** por precisión pura. En este panel pierde contra un baseline
  sin ajustar que cuesta 0.001 s.

### `PatchTSTForecaster`

- **Supuesto:** transformer sobre parches de la serie (no paso a paso),
  pensado para contextos largos. `use_futr_exog=False` en el run de
  referencia — es el único de los tres modelos de `neuralforecast` que
  **no** recibió temperatura futura, y compite sin ella.
- **Coste:** 6.67 s de ajuste medio, 220 136 parámetros.
- **Resultado de referencia:** MASE 0.381 — sexto puesto, el mejor de los
  modelos profundos que no usaron exógenas privilegiadas.
- **Cuándo usarlo:** contextos largos, series con patrones que se repiten a
  escalas de tiempo variables, cuando no hay exógena futura fiable.
- **Cuándo no:** paneles pequeños con estructura estacional simple, donde
  MSTL o `seasonal_naive_168` ya capturan la mayor parte de la señal a una
  fracción del coste.

### `LSTMForecaster` (propio, `chronolab.models.torch`) †

- **Supuesto:** encoder-decoder LSTM con proyección directa multi-paso (no
  recursiva) — el error no se acumula a lo largo del horizonte, a cambio de
  que la cabeza crezca con `h`. `use_futr_exog=True` por defecto: recibió
  temperatura futura real.
- **Coste:** 55.76 s de ajuste medio — **el modelo más caro del
  leaderboard**, 67 304 parámetros.
- **Resultado de referencia:** MASE 0.428 — décimo puesto, empatado
  prácticamente con `seasonal_naive_168` (0.426) **pese a la temperatura
  futura real** y a ser el modelo más lento de ajustar de los 19.
- **Cuándo usarlo:** hoy, en este panel: en ningún caso frente a las
  alternativas — pierde en precisión y en coste contra prácticamente todo lo
  demás. Puede tener sentido como punto de partida para experimentar con
  arquitecturas propias (encoder/decoder custom, pérdidas no estándar) donde
  `neuralforecast` no ofrece el control necesario.
- **Cuándo no:** producción, tal como está configurado aquí.

---

## Modelo fundacional zero-shot (`chronolab.models.adapters.chronos`)

### `ChronosForecaster` (Chronos-Bolt-small, 47.7M parámetros)

- **Supuesto:** modelo pre-entrenado en un corpus masivo de series
  temporales heterogéneas; `fit` no ajusta nada — solo recorta el contexto y
  fija el cutoff. Univariado puro: `needs_futr_exog=False`,
  `uses_hist_exog=False`, `uses_static_exog=False` sin condicional, aunque el
  panel declare exógenas.
- **Coste:** en el orden de milisegundos de "ajuste" (no hay entrenamiento);
  el coste real es la carga del modelo (~20 s la primera vez, cacheada
  después) y la inferencia, que corre en CPU.
- **Limitación de cola conocida:** dentro de su rango nativo (`[0.1, 0.9]`)
  interpola cuantiles reales; fuera de él —los extremos `0.025`/`0.975` de la
  rejilla canónica del proyecto— **clampa** en vez de extrapolar, así que sus
  colas salen más estrechas que las de un modelo calibrado a esos niveles.
- **No incluido en el leaderboard de referencia de este documento.** El
  adaptador existe, está testeado (`tests/unit/models/test_chronos.py`), pero
  `scripts/run_deep_analysis.py` —el script que produjo la tabla de
  `README.md`— no lo instancia. Correrlo sobre el mismo panel y las mismas
  ventanas es la tarea pendiente antes de poder comparar zero-shot contra
  ajustado con esta metodología.
- **Cuándo usarlo:** series nuevas sin historia suficiente para ajustar nada
  propio, o como referencia rápida de "cuánto aporta ajustar" frente a no
  ajustar nada — es exactamente la pregunta que debería responder una vez
  incluido en el leaderboard.
- **Cuándo no:** cuando hace falta interpretabilidad, cuando hay exógenas que
  claramente ayudan (el modelo no puede usarlas por diseño) o cuando el
  presupuesto de latencia no admite CPU.

---

## Detectores de anomalías (`chronolab.anomaly`)

Los cuatro comparten formato de score —`-log10(p)` con `p` un p-valor
conformal contra un pool de calibración nunca visto por el detector durante
el ajuste (Laxhammar & Falkman, 2010)— y el mismo umbral binario
(`score ≥ -log10(α)`), lo que en teoría los hace comparables a `α` fijo. En
la práctica, dos de los cuatro no cumplen esa promesa — ver la columna "tasa
de marcado observada", medida sobre el propio artefacto a `α = 0.05`
(`docs/ANOMALY_FINDINGS.md`).

| Detector | Tasa de marcado observada | `α` nominal | Calibrado |
|---|--:|--:|---|
| `ConformalDetector` | 8.4 % | 5 % | Sí, aproximadamente |
| `MatrixProfileDetector` | 4.8 % | 5 % | Sí, pero por construcción algebraica trivial — ver abajo |
| `IsolationForestDetector` | **48.6 %** | 5 % | **No** |
| `AutoencoderDetector` | **53.5 %** | 5 % | **No** |

### `ConformalDetector`

- **Supuesto:** el residuo del modelo base, normalizado por la anchura del
  intervalo predicho (`r = max(l−y, y−u) / (u−l)`, CQR), es la magnitud de
  no conformidad. No ve la serie directamente — ve el error de un
  forecaster ya ajustado.
- **Terreno donde funciona (`range_recall`):** escalón 0.98, pico 0.81 — un
  desplazamiento sostenido o un salto de un paso es exactamente un residuo
  sostenido o grande, y ambos con retardo mediano de **0 pasos**.
- **Punto ciego:** varianza (0.22), congelado (0.13), desfase estacional
  (0.03) — los tres producen residuos **intermitentes**, no sostenidos, y el
  detector solo marca los trozos donde el residuo se sale del intervalo.
- **Cuándo usarlo:** cuando ya existe un modelo base de forecasting
  razonable y las anomalías esperadas son desplazamientos de nivel o picos —
  el caso más común en monitorización operativa.
- **Cuándo no:** cambios de varianza o de fase estacional sin desplazamiento
  de nivel — necesita un detector que mire la forma, no el residuo puntual
  (`IsolationForestDetector` o `AutoencoderDetector`, una vez recalibrados —
  ver abajo).

### `MatrixProfileDetector`

- **Supuesto:** un discord (subsecuencia z-normalizada más distinta a su
  vecino más cercano) es una anomalía. Sin ajuste — `needs_calibration=False`
  literal, es el ejemplo que nombra el protocolo para "sin calibración".
- **Por qué su tasa de marcado cae cerca de `α` sin estar calibrado de
  verdad:** el pool de referencia es el **propio tramo que se está
  puntuando** (`_self_referential_score(pool, pool)`), lo que fija
  algebraicamente la fracción de puntos que cruzan el umbral en, ~`α`,
  independientemente de si hay o no anomalías reales. No es evidencia de
  buena calibración — es una identidad matemática del método.
- **Resultado de referencia:** `auc_pr = 0.175`, exactamente igual a la
  prevalencia — es la definición de estar al azar en este experimento. Solo
  ve picos (`range_recall` 0.19) por deformar la forma local; escalón y
  varianza son **invisibles por construcción** — la z-normalización elimina
  exactamente un cambio de media o de escala puros.
- **Cuándo usarlo:** referencia de contraste ("¿supera mi detector calibrado
  a algo sin ningún ajuste?"), o exploración inicial sin datos de
  calibración disponibles.
- **Cuándo no:** producción — no tiene garantía de tasa de falsos positivos
  real, y en este experimento queda al nivel del azar en AUC-PR.

### `IsolationForestDetector` — **no calibrado en este experimento**

- **Supuesto:** siete features retrospectivas por punto (valor crudo,
  residuo, z-score móvil, primera y segunda derivada, energía espectral
  local, hora del día); un `IsolationForest` aísla los puntos raros en ese
  espacio.
- **Por qué falla la calibración aquí:** `valor` es una feature **cruda**, y
  la serie sintética tiene tendencia. El pool de calibración sale de `dev`
  y el tramo puntuado es posterior — el nivel entero del holdout queda por
  encima de todo lo que el bosque vio al ajustarse, y le parece anómalo casi
  todo. Síntoma medible: `false_alarms_per_1000 = 37.3`, el peor de los
  cuatro por un factor de 3 a 14.
- **Lo que sí funciona pese a esto:** su **ranking** (no su umbral) es el
  mejor de los cuatro — `auc_pr = 0.571`, primero de la tabla. Discriminación
  y calibración son propiedades distintas, y aquí se separan de forma
  limpia.
- **Cuándo usarlo:** solo tras corregir la feature de tendencia (sustituir
  `valor` por el residuo desestacionalizado o una diferencia), y con
  evidencia de que el pool de calibración es representativo del tramo a
  puntuar.
- **Cuándo no:** tal como está configurado en este repositorio hoy — su
  recall de 0.89 en la tabla de `ANOMALY_FINDINGS.md` es marcar casi la
  mitad de la serie, no detección.

### `AutoencoderDetector` — **no calibrado en este experimento**

- **Supuesto:** un LSTM-autoencoder pequeño reconstruye ventanas de 24 pasos
  de la objetivo escalada; el error de reconstrucción es la magnitud de no
  conformidad. Entrena solo sobre tramos sin z-score robusto extremo
  (`trim_z`), para no aprender a reconstruir también la anomalía.
- **Por qué falla la calibración aquí:** el **49.4 %** de sus scores están
  exactamente en el techo de saturación del p-valor conformal
  (`log10(n_pool+1) ≈ 2.96`) — el error de reconstrucción de casi cualquier
  ventana del holdout supera todo el pool de calibración. Dentro de esa
  mitad no puede ordenar nada; es lo que la columna `severity` (magnitud sin
  acotar) existe para resolver, y lo que hunde su `auc_pr` a 0.306 pese a un
  `range_recall` de 0.90.
- **Cuándo usarlo:** anomalías de forma (cambio de varianza, desfase
  estacional) donde el detector conformal es ciego — su recall en esos tipos
  es el más alto de los cuatro (1.00 en varianza y desfase, según
  `docs/ANOMALY_FINDINGS.md` §3) — pero solo tras aumentar el pool de
  calibración o revisar el escalado para que el error deje de saturar.
- **Cuándo no:** sin corregir la saturación — la mitad de su rango dinámico
  no aporta información para ordenar severidad.

---

## Resumen: qué elegir según el escenario

| Escenario | Elección sugerida | Por qué |
|---|---|---|
| Referencia obligatoria, cualquier serie estacional | `SeasonalNaiveForecaster` | Es también el denominador de MASE; si nada lo bate, no hay nada que desplegar |
| Doble estacionalidad, sin exógenas fiables | `MSTLForecaster` | Mejor MASE del grupo sin presciencia, coste moderado |
| Exógena con pronóstico real disponible | `ProphetForecaster` (con `SimulatedForecastProvider`, no `Realized`) | Barato, interpretable, diseñado para regresores |
| Gradient boosting | `LightGBMForecaster(strategy="recursive")` | Mismo MASE que `direct` a 1/11 del coste en este panel |
| Interpretabilidad nativa por variable/instante requerida | `TFTForecaster` | Único con selección de variables y atención expuestas — a pesar de perder en precisión pura |
| Serie nueva sin historia para ajustar | `ChronosForecaster` | Zero-shot; falta incluirlo en el leaderboard compartido antes de confiar en el número |
| Monitorización de desplazamientos de nivel/picos | `ConformalDetector` | Calibrado de verdad, retardo de detección ~0 en su terreno |
| Monitorización de cambios de forma (varianza, fase) | `AutoencoderDetector`, tras recalibrar | Mejor recall en esos tipos, pero necesita arreglar la saturación antes de confiar en el umbral |
