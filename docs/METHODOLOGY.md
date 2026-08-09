# Metodología

Protocolo de backtesting completo, definición formal de cada métrica y
justificación de las decisiones de diseño de evaluación. Este documento es la
referencia normativa de `chronolab.evaluation` y `chronolab.anomaly`; el
código es la fuente de verdad y este documento debe leerse junto a él, no en
su lugar — cada sección enlaza al módulo correspondiente.

---

## 1. Protocolo de backtesting

### 1.1 Origen rodante: `Window` y `RollingOriginSplitter`

Una ventana (`chronolab.evaluation.splitters.Window`) es una tupla inmutable
y autoconsistente:

```
train_start ≤ cutoff < first_pred ≤ last_pred
first_pred  = cutoff + (gap + 1) · freq
last_pred   = first_pred + (h − 1) · freq
```

- **`cutoff`** es la frontera de información: todo lo que un modelo puede ver
  para esa ventana está en `[train_start, cutoff]`.
- **`gap`** son los pasos descartados entre el cutoff y la primera
  predicción. Sirve para emular latencia de datos y para cortar la
  autocorrelación de corto alcance en el punto de evaluación.
- **`h`** es el horizonte, en pasos.

`RollingOriginSplitter` es el **único emisor de particiones** del proyecto:
no existe ninguna función que acepte máscaras booleanas, índices arbitrarios
ni fechas sueltas para partir un panel. Genera las ventanas por aritmética
sobre la rejilla regular del panel (`panel.grid()`), ancladas al **final** del
panel — la última ventana evalúa exactamente hasta la última marca
disponible, y los cutoffs anteriores se obtienen restando `step_size`:

```python
cutoff_idx = last_cutoff_idx - (n_windows - 1 - planned_id) * step_size
```

Dos modos de entrenamiento:

- **`expanding`** (por defecto): `train_start` es siempre el inicio del
  panel; el entrenamiento crece con cada ventana.
- **`sliding`**: `train_start = cutoff − train_size + 1`; longitud de
  entrenamiento fija.

Ventanas cuyo entrenamiento no alcanza `min_context` pasos se descartan (las
más antiguas primero) con `ShortTrainWarning`, nunca se recortan a la fuerza.

### 1.2 Dev vs. holdout

`holdout_windows` marca los últimos N cutoffs del **plan** —no de las
ventanas supervivientes— como `stage="holdout"`; el resto son `"dev"`. La
distinción sobrevive a que una ventana antigua se descarte por historia
corta, porque se decide sobre la numeración del plan, no sobre la lista final.

`chronolab.evaluation.tuning.dev_only_panel` recorta el panel al tramo que
cubren exactamente las ventanas `dev` **antes** de que exista ningún trial de
Optuna: el `Forecaster` que construye cada trial no tiene forma física de leer
nada del holdout, porque los datos de holdout no están en la estructura que
recibe `backtest()` dentro del bucle de tuning. Es la barrera de "ausencia
física" (§2), aplicada al tuning de hiperparámetros.

### 1.3 Política de refit

Reajustar un modelo en cada ventana es correcto pero caro; reutilizar un
ajuste anterior es barato pero registra una decisión que cambia el resultado.
`BacktestPlan.refit_every_for` resuelve la tensión:

- `refit_every` explícito del plan, si se fija.
- Si no: `1` (reajuste en cada ventana) para modelos `refit_cost="free"` o
  `"cheap"`; `n_windows` (un único ajuste, en la ventana más antigua) para
  `refit_cost="expensive"`.

Reutilizar un ajuste **no es fuga** — el ajuste siempre es anterior a la
ventana que predice, comprobado por `_assert_fitted_at_or_before_cutoff` — pero
cambia el resultado, y por eso `refit` y `refit_every` se registran en
`model_runs` por cada par `(modelo, ventana)`.

Modelos con `predict` sensible al cutoff exacto (las tres redes de
`neuralforecast` y el LSTM propio, que emiten literalmente los `h` pasos
siguientes a su cutoff) exigen `refit_every=1`: reutilizar su ajuste en una
ventana posterior desalinearía la predicción, y el adaptador lo detecta y
falla en vez de alinear en silencio.

### 1.4 Las cuatro garantías anti-fuga del motor

Del docstring de `chronolab.evaluation.backtest`, con la comprobación exacta:

| # | Garantía | Mecanismo | Función |
|---|---|---|---|
| 1 | El modelo nunca ve el panel completo | Constructor único (`panel.train(window)` recorta a `ds ≤ cutoff`) + aserción | `_assert_train_at_cutoff` |
| 2 | Las exógenas históricas se cortan en el cutoff | Ausencia física — `FutrFrame` solo contiene columnas `futr_exog` | `_assert_futr_frame` |
| 3 | Nada de lo predicho cae en el pasado conocido | Aserción en el único camino que produce predicciones | `_assert_prediction_after_cutoff` |
| 4 | Un ajuste reutilizado no se adelanta a su ventana | Aserción antes de `predict` | `_assert_fitted_at_or_before_cutoff` |

Una violación de cualquiera de las cuatro lanza `LeakageError` o
`CutoffViolation`, y **se re-lanza sin capturar**: a diferencia de un fallo de
modelo (que ocupa una fila `status="failed"` en `model_runs` y el run
continúa), una fuga detiene el run entero. Un run con fuga no produce
resultados publicables, así que no produce ninguno.

### 1.5 El contrato de predicción

Toda predicción se valida antes de entrar en la tabla `forecasts`
(`_validate_prediction`):

- Cubre exactamente las series del entrenamiento de la ventana, ni más ni
  menos.
- Cae exactamente en `[first_pred, last_pred]` a la frecuencia del panel — ni
  un instante fuera de rejilla.
- Sin duplicados por `(unique_id, ds)`.
- El número de filas es exactamente `n_series × h`.

Un modelo que devuelve de más, de menos o descolocado no entra en la tabla —
falla con `PredictionContractError` — en vez de que un `merge` posterior lo
convierta en `NaN` silenciosos que parecen huecos del panel.

### 1.6 Exógenas futuras: `FutrProvider` y el vintage

Una exógena futura no es una columna del panel: es una función
`(as_of, ds) → valor`, porque su valor depende de **cuándo** se pregunta, no
solo de **cuándo** ocurre. `chronolab.data.futr.FutrProvider` formaliza eso
con tres implementaciones de honestidad decreciente:

- **`ArchivedForecastProvider`** — el pronóstico que de verdad estaba
  archivado en el momento del cutoff. La única honesta para publicar un
  resultado sin matiz.
- **`SimulatedForecastProvider`** — un pronóstico degradado con error
  realista sobre el valor observado.
- **`RealizedFutrProvider`** — el valor **realizado**, presciencia perfecta.
  Emite `PerfectForesightWarning` en construcción, siempre: el resultado de
  un run con este proveedor es una **cota superior**, nunca una estimación de
  rendimiento en producción, y solo es publicable etiquetado como tal.

`BacktestResult.futr_vintage` viaja pegado al resultado precisamente para que
comparar filas de vintages distintos dentro de un mismo leaderboard sea una
operación detectable, no un descuido silencioso.

### 1.7 Contraste estadístico: Diebold-Mariano

Comparar dos MASE de 0.28 y 0.29 sobre 144 observaciones sin contraste es
publicar ruido ordenado. `chronolab.evaluation.stats_tests.diebold_mariano`
implementa el test clásico con la corrección de muestra finita de Harvey,
Leybourne & Newbold (1997, "HLN"), sobre la diferencia de pérdidas
(`d_t = L(e_{1,t}) − L(e_{2,t})`) sin asumir independencia — corrige la
varianza con un estimador HAC (`hac_lag`) para el solape que introduce
`step_size < h`. `adjust_p_values` aplica corrección por comparaciones
múltiples (Benjamini-Hochberg) cuando se testean todas las parejas de un
leaderboard a la vez.

---

## 2. Métricas de forecasting: definición formal

Convenio común a todas: los pares con `NaN` en el valor observado o el
predicho se **descartan**, nunca se imputan — el panel conserva sus huecos
como `NaN` explícito, y una métrica no es el sitio para inventar el dato que
falta. `n_obs` viaja aparte y es lo que delata a un modelo evaluado sobre la
mitad de los puntos.

### 2.1 Métricas puntuales

**MAE**
```
MAE = (1/n) Σ |y_i − ŷ_i|
```

**RMSE**
```
RMSE = √[(1/n) Σ (y_i − ŷ_i)²]
```
Penaliza los errores grandes más que MAE; ordena los modelos de otra manera
cuando hay picos. Se reportan las dos.

**MAPE** (informativa, nunca de selección)
```
MAPE = (100/n) Σ |y_i − ŷ_i| / |y_i|
```
Indefinida en `y = 0` (esas observaciones se descartan); inestable cerca de
cero (`|y| < 10⁻³ · mean(|y|)` dispara `UnstableMetricWarning`); y sesgada
hacia modelos que infrapredicen — el error relativo máximo de predecir de
menos está acotado por el 100 %, el de predecir de más no. Es un sesgo del
criterio, no del modelo, y no deja rastro en un leaderboard ordenado por
MAPE.

**sMAPE** (versión M4, rango `[0, 200]`)
```
sMAPE = (100/n) Σ 2|y_i − ŷ_i| / (|y_i| + |ŷ_i|)
```
Circulan al menos tres definiciones que difieren en un factor 2 y en si el
denominador lleva valor absoluto; esta es la de la competición M4
(Makridakis et al. 2020). Números calculados con una definición distinta no
son comparables con estos.

### 2.2 MASE — la métrica principal

```
MASE = (1/n_test) Σ |y_i − ŷ_i| / q
```

con `q` el MAE del **naive estacional sobre el entrenamiento de la ventana en
curso**:

```
q = (1 / (n_train − m)) Σ_{t=m+1}^{n_train} |Y_t − Y_{t−m}|
```

`Y` es la serie del **tramo `[train_start, cutoff]` de esa ventana**, `m` es
`PanelSpec.mase_season` (la estacionalidad más corta declarada).

**Por qué MASE y no MAPE, en cuatro propiedades que MAPE no tiene**
(Hyndman & Koehler 2006):

1. Adimensional y comparable entre series de escalas distintas (30 kW y
   3000 kW producen MAE incomparables; MASE divide por un error de
   referencia medido en la propia serie).
2. Definida cuando la serie pasa por cero (demanda nocturna, generación
   solar) — MAPE no lo está.
3. Simétrica: penaliza igual por arriba y por abajo, sin el sesgo de MAPE
   hacia la infrapredicción.
4. Cero natural interpretable: `MASE < 1` significa "bate al naive
   estacional sobre la escala de su propio entrenamiento".

**Dónde se rompe casi siempre esto en otros repositorios, y cómo se evita
aquí.** El denominador se calcula sobre el conjunto de *test*, o sobre la
serie *completa*, o una sola vez para todo el run. Las tres variantes son
fuga — el denominador incorpora información posterior al cutoff — y además
destruyen la comparabilidad, porque el mismo modelo cambiaría de MASE según
qué ventana se evalúe. Aquí `mase_denominators` recorta el panel a
`[train_start, cutoff]` de **cada ventana** y calcula `q` serie a serie; se
persiste, así que es auditable por un tercero sin volver a correr el modelo.

**Regla de agregación: nunca se promedia un promedio.** La fila "modelo X
sobre todas las series" no es el promedio de las filas por serie: es el
mismo cálculo repetido sobre el conjunto crudo de observaciones. Con MAE da
igual si todas las series tienen el mismo número de puntos; con MASE, sMAPE,
MAPE y la cobertura, no. El vector de denominadores se escala fila a fila
antes de promediar — nunca se promedian los errores y se divide por un `q`
medio, que mezclaría escalas y no sería MASE de nada.

### 2.3 Métricas probabilísticas

**Pérdida pinball** de un cuantil `τ`:
```
PLτ(y, q̂) = τ(y − q̂)        si y ≥ q̂
           = (1−τ)(q̂ − y)    si y < q̂
```
Es la regla de puntuación propia del cuantil (Gneiting & Raftery 2007): se
minimiza en expectativa cuando `q̂` es el cuantil `τ` verdadero. Un modelo no
puede mejorarla ensanchando o estrechando sus intervalos a conveniencia — a
diferencia de mirar solo la cobertura.

**Cobertura empírica**
```
coverage = (1/n) Σ 1[lower_i ≤ y_i ≤ upper_i]
```
Se compara siempre contra la **cobertura nominal** del intervalo: un
intervalo al 95 % que cubre el 78 % está mal calibrado, y esa diferencia es
lo que hay que reportar. Viaja siempre junto a `interval_width` — un
intervalo de anchura infinita cubre el 100 % y no dice nada.

**CRPS discreto** — aproximación por integración trapezoidal de la pinball
sobre la rejilla de cuantiles:
```
CRPS(F, y) = 2 ∫₀¹ PLτ(y, F⁻¹(τ)) dτ  ≈  trapecio sobre {τ}
```
Con la rejilla canónica `[0.025, 0.975]` queda fuera un 5 % de masa en las
colas: el valor es una **cota inferior** del CRPS verdadero, y no es
comparable entre rejillas distintas.

---

## 3. Métricas de anomalías: por qué no point-adjusted F1

### 3.1 El vicio que se rechaza, con números

El *point-adjusted F1* (F1-PA) marca como detectado un segmento entero de
verdad si el detector acierta **un solo** punto suyo:

```python
adjusted = predicted.copy()
for start, end in true_ranges:
    if predicted[start:end+1].any():
        adjusted[start:end+1] = True
f1_pa = point_f1(adjusted, actual)
```

Kim et al. (2022, "Towards a Rigorous Evaluation of Time-series Anomaly
Detection") y Paparrizos et al. (2022, TSB-UAD/VUS) muestran que bajo esta
regla un detector que emite **ruido aleatorio** obtiene F1 comparable al
estado del arte: basta con salpicar puntos al azar para tocar todos los
segmentos largos. `tests/unit/evaluation/test_anomaly_metrics.py` reproduce
la inflación de forma cuantitativa sobre los mismos datos del proyecto:

- ruido aleatorio → F1-PA > 0.55
- ese mismo F1-PA es más de **5 veces** el F1 honesto (no ajustado) sobre las
  mismas etiquetas.

Una métrica bajo la que el ruido gana no mide nada, y este módulo no la
implementa en ningún punto del código.

### 3.2 Cinco familias que fallan de formas distintas

La alternativa no es "una métrica mejor": es que un detector solo parezca
bueno si lo es bajo criterios que no comparten sesgo.

**Por rangos** (Tatbul et al. 2018) — trata cada anomalía como un intervalo,
no como puntos sueltos:

```
Recallrange(R) = Σ_{Rᵢ ∈ R} ω(Rᵢ, P) · [α · ExistenceReward(Rᵢ, P)
                  + (1−α) · OverlapReward(Rᵢ, P)]
```

con `ω` un sesgo posicional (`flat`/`front`/`back`/`middle` — dónde dentro
del rango pesa más acertar) y `OverlapReward` penalizado por un factor de
cardinalidad (`"reciprocal"`: dividir un evento en 5 detecciones vale un
quinto) — es precisamente la puerta por la que el point-adjusted vuelve a
colarse si la cardinalidad no se penaliza (`cardinality="one"`).

**De afiliación** (Huet et al. 2022) — mide **distancia**, no solape: una
detección que llega 2 pasos tarde no vale lo mismo que una que no llega. Se
calibra contra el azar — **0.5 es el valor esperado de una predicción
uniforme**, no 0 — y por eso siempre se reporta junto a la de rangos, para no
leer 0.5 con la intuición de una precisión clásica.

**VUS-PR** (Paparrizos et al. 2022) — integra la superficie precisión-recall
sobre el umbral **y** sobre una tolerancia de desalineamiento temporal a la
vez, evitando fijar ninguno de los dos a mano.

**AUC-PR puntual** — la referencia mínima, sin ninguna noción de rango. Se
incluye a propósito para poder decir cuánto cambia la conclusión al pasar a
métricas por rango: si no cambiara, el resto del módulo sobraría (y en el
experimento de referencia del proyecto, sí cambia — ver
`docs/ANOMALY_FINDINGS.md` §2).

**Operativas** (`detection_delay`, `false_alarm_rate`) — lo que decide si un
detector se despliega: cuánto tarda en avisar y cuántas veces avisa en falso,
a nivel de **evento**, no de punto (un operador se avisa una vez por
incidente).

### 3.3 Reglas de agregación

- **La máscara `scorable` se interseca antes de comparar**
  (`common_scorable_mask`): un detector de ventana 512 puntúa menos
  instantes que uno de ventana 1, y evaluar a cada uno sobre su propio
  soporte favorecería al de ventana larga por haberse saltado el arranque de
  la serie — su parte peor condicionada.
- **Lo que es una media sobre rangos se agrega juntando rangos**, no
  promediando medias — la misma regla de §2.2 aplicada a anomalías.
- **AUC-PR y VUS-PR se promedian entre series, no se agrupan** — el score es
  ordinal solo *dentro* de un par (detector, serie); agrupar rankings de
  series distintas produciría un número que parece un AUC y no lo es.

### 3.4 Umbralización

Separar puntuar (score continuo) de umbralizar (etiqueta binaria) es
obligatorio: VUS-PR necesita el score continuo, F1 por rangos necesita
etiquetas. Si el detector devolviera directamente etiquetas se perdería
irreversiblemente lo que exige la métrica de rango. `ConformalThresholder`
resuelve el umbral en forma cerrada (`-log10(α)`) para los detectores basados
en p-valor conformal, y marca `reachable=False` cuando el umbral pedido cae
por debajo de la resolución del pool de calibración (`1/(n+1)`), en vez de
devolver un número que ningún punto podrá cruzar nunca sin que nada lo
advierta.

---

## 4. Referencias

Ver la lista completa en [`README.md`](../README.md#referencias). Las
directamente citadas en este documento:

- Hyndman & Koehler (2006) — MASE.
- Makridakis et al. (2020) — sMAPE (M4).
- Gneiting & Raftery (2007) — pinball / CRPS como reglas de puntuación propias.
- Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997) — contraste
  de precisión predictiva y su corrección de muestra finita.
- Tatbul et al. (2018) — precisión/recall por rangos.
- Huet, Navarro & Rossi (2022) — precisión/recall de afiliación.
- Paparrizos et al. (2022) — VUS-PR.
- Kim et al. (2022) — crítica del point-adjusted F1.
