# Detector conformal de anomalías — diseño

Estado: **aprobado e implementado**. Las secciones marcadas con "la implementación
obligó a corregir" recogen lo que se descubrió al construirlo; el resto es el diseño
aprobado, sin retocar.

Detector principal de `chronolab.anomaly.conformal`: una observación es anómala cuando
cae fuera del intervalo de predicción conformalizado del mejor modelo del run. La tasa de
falsos positivos queda acotada por construcción, sin asumir normalidad de los residuos.

Este documento resuelve los cinco puntos del bloque de revisión, fija las firmas públicas
y enumera las **enmiendas al contrato existente** que la implementación necesitó.

---

## 0. Alcance y reparto por módulo

El detector no cabe entero en `conformal.py`, y forzarlo rompería límites que
`docs/ARCHITECTURE.md` ya fijó. Reparto:

| Módulo | Responsabilidad | Estado |
|---|---|---|
| `anomaly/conformal.py` | No conformidad, grupos Mondrian, estado en línea, `score` y `severity` | Implementado |
| `anomaly/thresholds.py` | `ConformalThresholder`: tabla α → umbral | Implementado (trivial, ver §3.4) |
| `anomaly/events.py` | Agregación en eventos y emparejamiento con la verdad | Implementado (punto 4) |
| `artifacts/reader.py` | `scoring_frame(...)`, único constructor de `ScoringFrame` | Implementado en memoria, ver §0.1 |

`models/wrappers.py::ConformalWrapper` (D22) es **otra cosa** y no se toca: produce
intervalos dentro de `fit` de un modelo; este detector decide qué cae fuera de ellos.

### 0.1 Dependencia bloqueante

`artifacts/reader.py` era un docstring. Como `ScoringFrame` solo puede construirlo el
lector (D12), el detector no era ejecutable de extremo a extremo sin él. Se implementó el
alcance mínimo:

```python
def scoring_frame(result: BacktestResult, *, model_id: ModelId, stage: Stage) -> ScoringFrame
```

es decir, la variante **en memoria**, desde un `BacktestResult`. La ruta desde parquet
queda para la tarea de artefactos: lo que cambiará es de dónde salen las tablas, no cómo se
recorta el tramo. Alternativa —construir `ScoringFrame` en los tests— descartada: abriría
un segundo constructor y anularía la barrera de D12. `BacktestResult` se importa bajo
`TYPE_CHECKING`, con el mismo patrón que el motor usa para `FutrProvider`, de modo que
`artifacts` no depende de `evaluation` en tiempo de ejecución.

---

## 1. Partición train / calibración / test

### 1.1 De dónde salen los residuos

El detector no parte datos: **consume los residuos fuera de muestra que el motor de
backtesting ya produjo**. Cada fila de `forecasts` es un par `(y, y_hat)` donde `y_hat`
vino de un modelo ajustado con `ds <= cutoff` y `ds` de la predicción es posterior al
cutoff, garantizado estructuralmente por `_assert_train_at_cutoff` y
`_assert_prediction_after_cutoff` en [backtest.py](src/chronolab/evaluation/backtest.py).
La calibración no puede solaparse con el entrenamiento del forecaster porque **no hay
ningún residuo dentro de muestra en la tabla de la que se lee**.

### 1.2 Dónde cae el corte

La frontera calibración/test es exactamente la frontera `dev` / `holdout` que el
splitter ya emite:

```
ventanas dev                                   ventanas holdout
w0      w1      w2      w3      w4      w5  |  w6      w7
[cal ][cal ][cal ][cal ][cal ][cal ]        |  [test][test]
                                    calib.end ^  ^ frame.start
```

- `calib  = scoring_frame(result, model_id=m, stage="dev")`
- `frame  = scoring_frame(result, model_id=m, stage="holdout")`

Tres propiedades salen gratis de reutilizar `Window.stage` en lugar de inventar un
parámetro nuevo:

1. **El corte es a granularidad de ventana, nunca dentro de una.** Partir dentro de una
   ventana dejaría residuos del mismo origen de predicción a ambos lados: comparten
   ajuste, están fuertemente correlados, y el cuantil de calibración se estimaría en parte
   con la misma predicción que después se puntúa. No es fuga de futuro, pero rompe la
   intercambiabilidad de la que cuelga la cota de cobertura.
2. **`stage="holdout"` son siempre las últimas ventanas del plan** (`_stage` decide sobre
   la numeración del plan, y los cutoffs crecen), así que `calib.end < frame.start` sin
   comprobar nada.
3. **La disciplina de tuning se hereda.** γ, el tamaño de pool, los bins y α_nominal se
   eligen sobre `dev`; el holdout solo se reporta.

Si el plan tiene `step_size < h`, los tramos evaluados de dev y holdout se solapan,
`calib.end >= frame.start` y `score` lanza `CutoffViolation`. **No hay que añadir ninguna
comprobación**: la aserción del protocolo ya es la barrera correcta, y el fallo es ruidoso
en lugar de silencioso.

### 1.3 Qué le exigimos al plan del run de anomalías

No se toca `BacktestPlan`. Se valida en el punto de entrada del run de detección:

| Parámetro | Exigencia | Motivo |
|---|---|---|
| `step_size == h` | obligatoria | Teselado: cada `ds` del tramo evaluado aparece **exactamente una vez**, con `h_step` de 1 a `h`. Es lo que da rejilla completa y una fila por `(unique_id, ds)`, que es lo que el protocolo de `score` exige. |
| `holdout_windows >= 1` | obligatoria | Sin holdout no hay tramo que puntuar. |
| `mode="sliding"` | recomendada | Ver §1.4. |
| `refit_every == 1` | recomendada | Ver §1.4. |

### 1.4 Dos sesgos que hay que nombrar, no esconder

- **Modo expansivo.** El modelo de la ventana 0 se entrenó con mucha menos historia que el
  de la ventana 7. Sus residuos son sistemáticamente mayores, luego el cuantil de
  calibración sale ancho y el detector **subdetecta**. Es un sesgo conservador, pero es un
  sesgo. `mode="sliding"` iguala la longitud de entrenamiento de todas las ventanas y hace
  los residuos mucho más próximos a idénticamente distribuidos. Es la razón técnica por la
  que el run de anomalías debería ser deslizante aunque el de forecasting no lo sea.
- **`refit_every > 1`.** Dentro de un bloque de refit los residuos comparten ajuste y
  empeoran según envejece el modelo: intervalo demasiado ancho al principio del bloque y
  demasiado estrecho al final. Condicionar por `windows_since_fit` lo arreglaría, pero
  multiplica los grupos de §5 sin presupuesto muestral para ello. Decisión: exigir
  `refit_every=1` de facto por recomendación y **registrar** la política aplicada; si
  alguien corre con refit perezoso, el informe de cobertura de §6 lo delata.

El corrector adaptativo en línea de §2 absorbe automáticamente ambos sesgos, lo que es
un argumento adicional para que el modo adaptativo sea el predeterminado.

---

## 2. Split vs adaptativo en línea

### 2.1 Una sola implementación, tres regímenes

El detector mantiene, por grupo (§5):

- un **pool de calibración** de las últimas `pool_size` no conformidades, FIFO;
- un **estado ACI** de nivel efectivo `α_eff` por cada α de la rejilla.

Y entonces:

| `pool_size` | `gamma` | Método resultante |
|---|---|---|
| `None` (ilimitado) | `0.0` | **Split conformal exacto** — la línea base de la comparación |
| `K` finito | `0.0` | Conformal de ventana rodante (cuantil actualizado al llegar observaciones) |
| `K` finito | `> 0` | **Adaptativo en línea (ACI + pool rodante)** — predeterminado |

Que los casos degenerados sean *exactamente* los métodos clásicos es deliberado: convierte
"split frente a adaptativo" en un experimento con dos escalares, no en dos rutas de código
que hay que verificar por separado.

### 2.2 El trade-off, sin adornos

**Split conformal** da cobertura marginal `>= 1 - α` en muestra finita, con la única
hipótesis de intercambiabilidad. El cuantil es fijo, auditable y reproducible. Su problema
es que la intercambiabilidad se rompe exactamente cuando importa: cambio de régimen,
deriva de la varianza del residuo, degradación del modelo. Cuando se rompe, el detector
falla de forma silenciosa y bimodal —inunda de falsos positivos, o se queda ciego— y la
garantía teórica sigue impresa en el informe.

**ACI** (`α_eff ← α_eff + γ·(α − err)`, con `err = 1{fuera del intervalo}`) sustituye la
garantía de muestra finita por una de **tasa empírica a largo plazo**, que se cumple *sin*
intercambiabilidad y bajo deriva arbitraria: la desviación de la tasa observada respecto de
α está acotada por `(α_1 + γ) / (T·γ)`. Lo que se paga:

1. **No hay garantía local.** En cualquier tramo corto la cobertura puede estar muy lejos
   de 1−α; solo la media larga converge.
2. **γ es un hiperparámetro real.** γ grande sigue la deriva pero oscila y produce
   umbrales inestables; γ pequeño es estable pero llega tarde. Se ajusta en `dev`, nunca
   en holdout. Predeterminado γ = 0.01 para datos horarios.
3. **`α_eff` se sale del intervalo.** Con `α_eff <= 0` el intervalo es infinito (no marca
   nunca) y con `α_eff >= 1` es vacío (marca siempre). Esas excursiones **son** el
   mecanismo de compensación. Se recortan a `[1e-4, 0.5]` y se **cuentan**: una excursión
   sostenida es el síntoma de que el modelo base se ha roto, no un detalle numérico.
4. **El detector pasa a ser dependiente del orden.** Se compensa en §2.5.

### 2.3 Cuándo se puede actualizar: la trampa del adelanto

El error habitual es actualizar con `y_{t-1}` para puntuar `t`. Es incorrecto en cuanto
`lead > 1`: el intervalo forma parte de la **predicción**, y la predicción de `t` se emitió
en su origen `o = cutoff`, cuando `y` solo se conocía hasta `o`. Los residuos de
`(o, t)` no existían todavía.

Regla: **el estado usado para puntuar un punto es el estado en el origen de su
predicción**. Con `step_size = h`, todos los puntos de una ventana comparten origen, luego

> el estado avanza en las fronteras de ventana, no punto a punto.

Antes de puntuar la ventana de cutoff `o`, se ingieren todos los residuos con `ds <= o` no
ingeridos aún, en orden de `ds`, aplicando la recursión ACI secuencialmente. Es la
recursión original, liberada por bloques porque antes no había información que liberar.
Esto exige que `ScoringFrame` transporte `cutoff` (enmienda A, §7): sin él el origen habría
que reconstruirlo con aritmética sobre `gap` y `h_step`, que es justo donde nacen los
off-by-one.

### 2.4 La absorción de la anomalía

Este es el modo de fallo grave del ACI aplicado a detección, y casi nunca se menciona:
mientras una anomalía persiste, `err = 1`, luego `α_eff` baja, luego el intervalo se
ensancha, luego **el detector deja de marcar la anomalía que estaba detectando**. Un cambio
de nivel o una congelación de sensor se absorben como la nueva normalidad en unas decenas
de pasos. Lo mismo por la otra vía: si el residuo anómalo entra en el pool, sube el cuantil.

Cuarentena propuesta, con presupuesto acotado:

- Tras `freeze_after` marcas consecutivas (predeterminado 3) se **suspenden** la recursión
  ACI y la ingesta al pool.
- La suspensión dura como mucho `max_freeze` pasos (predeterminado `2 × spec.seasonalities[0]`,
  48 h). Pasado ese punto se reanuda todo: un desplazamiento permanente **debe** acabar
  siendo el nuevo normal, y que eso sea una política declarada y no un accidente es la
  diferencia entre un detector y un generador de alarmas perpetuas.
- Los puntos marcados se excluyen del pool con un presupuesto de `3α` del total ingerido
  por grupo. Superado el presupuesto se ingieren igualmente: excluir más que eso ya no es
  robustez, es negarse a ver un cambio de régimen.

Honestidad: la cuarentena rompe la garantía de tasa del ACI, que asume actualización en
todos los pasos. La garantía se conserva sobre la **subsucesión actualizada**, y el número
de pasos congelados se registra para que quien lea el número sepa sobre qué se cumple.

### 2.5 Reproducibilidad frente a estado

El protocolo exige objetos calibrados inmutables y `score` debe poder llamarse dos veces
con el mismo resultado. Solución, calcada de D5:

- `FittedConformalDetector` es `frozen`. `score(frame)` **copia** el estado, corre la
  recursión sobre la copia y la descarta. Puntuar dos veces da bit a bit lo mismo.
- Para el uso en línea real hay un método aparte,
  `advance(frame) -> FittedConformalDetector`, que devuelve un objeto **nuevo** con el
  estado avanzado y `cutoff = frame.end`. Igual que `fit` frente a mutar `self`: avanzar
  es explícito y queda en el tipo.

No hay aleatoriedad en ninguna parte del detector; el orden de proceso es
`(unique_id, ds)` ascendente. No necesita semilla.

---

## 3. Score continuo

### 3.1 No conformidad: CQR normalizada por anchura

Para un punto `t`, con el par de cuantiles del modelo `(l_t, u_t)` al nivel nominal
`1 - α_nom` (predeterminado `q_0250` / `q_9750`) y anchura `w_t = u_t - l_t`:

```
s_t = max(l_t - y_t,  y_t - u_t)          # < 0 dentro, 0 en el borde, > 0 fuera
r_t = s_t / w_t                            # adimensional
```

Se usa CQR (*conformalized quantile regression*) y no el residuo absoluto porque hereda la
asimetría y la heterocedasticidad que el modelo ya captura y solo corrige su
descalibración. §7.3 de la arquitectura dice explícitamente que la simetría es falsa en
demanda eléctrica; un score sobre `|y - y_hat|` la reintroduce por la puerta de atrás.

Dividir por `w_t` es la primera capa de la respuesta al punto 5: si el modelo predice
bandas más anchas en la punta de la tarde, `r_t` ya sale casi homoscedástico antes de
agrupar nada.

`scorable = False` (y `score = NaN`) si `y_t` es `NaN`, si el par de cuantiles es `NaN`
—modelo con `supports_quantiles=False`— o si `w_t <= 0`. Nunca se inventa un intervalo.

### 3.2 Cuantil conformal y umbral

Con el pool del grupo `g`, de tamaño `n_g`, ordenado ascendentemente:

```
Q_g(α) = r_(k)   con  k = ⌈(n_g + 1)(1 - α)⌉        exige  n_g >= ⌈1/α⌉ - 1
```

Intervalo conformalizado: `[l_t - Q_g(α)·w_t,  u_t + Q_g(α)·w_t]`, de anchura
`w_t^c = w_t·(1 + 2·Q_g(α))`.

### 3.3 Las dos columnas, y por qué son dos

**`score` — calibrado, ordinal, libre de α.** Es el p-valor conformal:

```
p_t   = (1 + #{i ∈ C_g : r_i >= r_t}) / (n_g + 1)
score = -log10(p_t)
```

Bajo intercambiabilidad dentro del grupo, `p_t` es super-uniforme, luego
`P(p_t <= α) <= α`: **la tasa de falsos positivos está acotada por construcción y sin
asumir normalidad**, que es la afirmación central del detector. Además se cumple la
identidad exacta

```
p_t <= α   ⟺   r_t > Q_g(α)   ⟺   y_t fuera del intervalo conformalizado al nivel α
```

(ambos lados equivalen a `#{r_i >= r_t} <= ⌊(n_g+1)α⌋ - 1`). Marcar por p-valor y marcar
por intervalo son literalmente la misma decisión.

**`severity` — magnitud legible, el punto 3 del bloque.** Cuánto se sale la observación
del intervalo, normalizado por la anchura del intervalo:

```
severity_t = max(l_t^c - y_t,  y_t - u_t^c) / w_t^c        con [l^c, u^c] al nivel α_ref
           = (r_t - Q_g(α_ref)) / (1 + 2·Q_g(α_ref))
```

Es una sola fórmula continua: `-0.5` en el centro exacto del intervalo, `0` en el borde,
`+1.0` cuando la observación se sale una anchura completa de intervalo. Se normaliza por la
anchura **conformalizada** y no por la del modelo: la del modelo puede estar descalibrada
—un modelo con bandas absurdamente estrechas mostraría severidades enormes en todas
partes—, mientras que la conformalizada es la escala empíricamente correcta.

La segunda igualdad demuestra que `severity` es una transformación afín estrictamente
creciente de `r_t` dentro del grupo, luego **las dos columnas no pueden discrepar sobre si
un punto está marcado**: `severity > 0 ⟺ score >= -log10(α_ref)`.

Por qué hacen falta las dos, y no es redundancia:

| | `score` | `severity` |
|---|---|---|
| Escala | p-valor, comparable entre grupos y series | anchuras de intervalo |
| Cota | **satura** en `log10(n_g + 1)` | no acotada |
| Sirve para | umbralizar con garantía, VUS-PR, curvas PR | ordenar dentro de la cola saturada, agregar eventos, leerlo un humano |

La saturación no es un defecto que se pueda arreglar: con `n_g` puntos de calibración no
existe información para distinguir severidades más allá del `1/(n_g+1)`. `severity` es
exactamente lo que ordena esa cola. Regla para `anomaly_metrics`: **ordenar por `score`,
desempatar por `severity`.**

Consecuencia operativa que hay que escribir en el informe: **la rejilla de α se trunca en
`1/(n_min + 1)`**. Con 200 puntos por grupo, α = 0.001 es inalcanzable y se marca como tal
en lugar de extrapolar una cola que no se ha observado.

### 3.4 La tabla de umbrales es trivial, y eso es la prueba

`ConformalThresholder` devuelve `threshold(α) = -log10(α)`, con `unique_id = null`, igual
para todo grupo y toda serie. Toda la calibración vive en el score. Que la tabla
`anomaly_thresholds` quede casi vacía para este detector no es un desperdicio: es la
comprobación visible de que el score sí está calibrado, frente a IsolationForest o el
autoencoder, cuyos umbrales serán cuantiles empíricos que hay que estimar.

Con ACI el umbral sigue siendo `-log10(α)` porque la corrección se pliega dentro del score.
Marcar al nivel α equivale a `p_estático ≤ α_eff(α)` —por la identidad de §3.3—, así que el
p-valor que hace comparable todo es el α que resuelve `α_eff(α) = p_estático`. El mapa
`α ↦ α_eff(α)` se **monotoniza por máximo acumulado** (las excursiones de ACI no garantizan
por sí solas que el conjunto marcado crezca con α) y se invierte por interpolación en escala
logarítmica. Coste: una recursión escalar por (grupo, α) y una inversión por punto.

Un detalle que la primera redacción de este documento tenía mal, y que la implementación
obligó a corregir: tomar `p̃ = min{α ∈ rejilla : marcado}` aplasta a un valor constante todo
lo que no se marca ni siquiera al α más grande de la rejilla, es decir la inmensa mayoría de
los puntos, y destruye la curva PR por debajo de ese α. **Fuera de la rejilla se extrapola
manteniendo la razón del extremo**, con lo que el score sigue siendo continuo en toda la
recta. Beneficio colateral: con `γ = 0` se tiene `α_eff ≡ α` y la inversión es la identidad
**exacta**, también fuera de la rejilla, de modo que split conformal cae como caso
degenerado exacto y no como aproximación. Hay un test que lo comprueba.

Alternativa descartada: umbral variable en el tiempo, que obligaría a añadir `ds` a
`anomaly_thresholds` y convertiría el deslizador de la app en un recálculo, prohibido por A5.

---

## 4. Agregación en eventos

Vive en `anomaly/events.py`. Un evento es una tirada maximal de puntos marcados, con
tolerancia declarada a huecos.

```python
def aggregate_events(scores, *, alpha, merge_gap=2, min_points=1) -> pd.DataFrame
def match_events(detected, truth, *, tol_steps=1) -> pd.DataFrame
```

**Fusión con tolerancia.** Dos tiradas separadas por `<= merge_gap` puntos no marcados se
fusionan. Una anomalía real casi nunca produce una tirada ininterrumpida: un solo punto que
vuelve a entrar en la banda parte un evento en dos y destroza la precisión a nivel de
evento. La tolerancia se declara y se registra; el predeterminado es 2 pasos.

`min_points = 1` por defecto: con α pequeño, el `spike` de un solo punto es precisamente
uno de los cinco tipos que `anomaly/injection.py` inyecta. Filtrarlo sería filtrar el
objetivo.

**Campos por evento** (los tres primeros son lo que pide el punto 4):

| Campo | Definición |
|---|---|
| `n_points` | puntos marcados |
| `duration_steps` | pasos de rejilla de `start_ds` a `end_ds`, ambos inclusive |
| `peak_score`, `peak_severity` | máximos dentro del evento |
| `cum_severity` | **Σ `severity` sobre los puntos marcados**; unidades: anchuras de intervalo × pasos, es decir el área fuera de la banda |
| `peak_ds` | instante del máximo |
| `direction` | `over` / `under` / `mixed` según el signo de la salida |

`n_points` y `duration_steps` difieren exactamente cuando hubo fusión, y esa diferencia es
un diagnóstico: dice si el evento fue sólido o intermitente. `direction` se separa porque
en demanda eléctrica un pico de consumo y una caída son incidentes operativamente
distintos, y colapsarlos pierde el bit más accionable. `cum_severity` es lo que ordena
"una hora muy fuera" frente a "seis horas ligeramente fuera", que `peak_severity` sola no
distingue.

`event_id` es determinista y reproducible:
`f"{detector_id}|{unique_id}|{alpha:.4f}|{start_ds:%Y%m%dT%H%M}"`. Incluye α porque los
eventos **son función de α**: sin ella, dos filas de la tabla con distinto α colisionarían.

**Emparejamiento uno a uno.** Un evento detectado empareja con uno real si se solapan al
menos un paso tras expandir el evento real `tol_steps` a cada lado (tolerancia de latencia
de detección). La asignación es **uno a uno**, voraz por solape descendente y desempate por
inicio más temprano. Es lo que impide reproducir el vicio del *point-adjusted F1* que la
arquitectura prohíbe explícitamente: si un evento real se parte en cinco detectados, el
resultado es **1 acierto y 4 falsas alarmas**, nunca 5 aciertos.

---

## 5. Heteroscedasticidad

Un umbral global sobre residuos crudos es un error: la varianza del residuo de demanda a
las 04:00 y a las 20:00 difiere en un factor grande. Un cuantil global da cobertura
*marginal* correcta —la tasa de falsos positivos es α en promedio— mientras que la tasa
condicional es quizá 0.1 % de madrugada y 12 % en la punta de la tarde. El detector
**parece** calibrado en el informe y es inútil en operación.

Tres capas, en orden de cuánto compran:

### 5.1 Que lo haga el modelo (§3.1)

`r_t = s_t / w_t` explota que un modelo con features horarias ya predice bandas más anchas
en las horas de más varianza. Conformal solo tiene que corregir la escala global, no la
forma. Es la capa más barata y la que más aporta si el modelo base es decente; no aporta
nada si el modelo emite bandas planas.

### 5.2 Conformal de Mondrian: la respuesta principal

Calibración **por grupo**, con un pool por grupo. Da cobertura *condicional al grupo*,
`P(marcar | g) <= α` para cada `g`, que es exactamente la garantía que se quiere.

Grupo: `g = (unique_id, hora local, bin de adelanto)`.

- **Hora local, no UTC.** El ciclo de demanda sigue el reloj local; el panel es UTC
  ingenuo (I2). Agrupar por hora UTC emborrona el grupo medio año en un país con DST. La
  conversión UTC → local es total y no ambigua (la ambigua es la inversa, que es por lo que
  I2 almacena UTC). Se reutiliza `data/calendar.py`, que la arquitectura declara **único
  módulo donde conviven UTC y hora local**; `anomaly/` no puede convertirse en el segundo.
  Requiere promover un `local_hour(...)` público allí (enmienda D).
- **Bin de adelanto.** El `lead = gap + h_step` es el otro determinante de primer orden de
  la escala del residuo. Con `step_size = h`, un mismo `ScoringFrame` contiene adelantos de
  1 a `h`: pooling ciego sobre ellos produce un intervalo demasiado ancho a adelanto 1 y
  demasiado estrecho a adelanto `h`. Bins predeterminados: `{1}`, `{2..6}`, `{7..24}`,
  `{>24}`.

**El presupuesto muestral es el límite real.** Un grupo necesita
`n_g >= ⌈1/α_min⌉ - 1` para que el cuantil exista. Cuentas concretas: un año horario son
8760 puntos por serie, ~6000 en ventanas `dev`. Con 24 horas × 4 bins de adelanto = 96
grupos salen ~60 puntos por grupo, insuficiente por debajo de α = 0.02. Con 6 bins horarios
× 4 de adelanto = 24 grupos salen ~250, que soporta α = 0.005.

Por eso el grupo no es fijo sino una **cadena de repliegue**, evaluada por punto contra el
tamaño de pool vigente:

```
(uid, hora, bin_lead) → (uid, bin_hora, bin_lead) → (uid, bin_hora) → (bin_hora) → global
```

Se usa el primer nivel con `n_g >= min_calib`. **El nivel usado y el `n_g` efectivo se
persisten por punto** (`calib_n`, enmienda B). Es el mismo principio que persistir el
denominador de MASE (D17): el número que sostiene la garantía tiene que ser auditable desde
la app, no una afirmación del documento. Nunca se extrapola una cola no observada.

### 5.2.1 La tensión que la implementación destapó: granularidad contra velocidad

Medir el diseño en marcha reveló una interacción entre §2 y §5 que no estaba escrita, y que
condiciona cómo se configura un run:

> **Cuanto más fino es el grupo, más lento se adapta el pool rodante**, porque cada grupo se
> refresca a `1/G` del ritmo de llegada de datos.

Con `G` grupos, un pool de tamaño `K` cubre `K·G` pasos de historia. Con 18 grupos y
`pool_size = 150` eso son ~2.700 pasos: más de cien ventanas de historia, y entonces el pool
no es rodante en ningún sentido útil. Medido sobre un run con cambio de régimen justo
después de la calibración (σ de 0.6 a 3.0, α = 0.05):

| configuración | tasa de marcado |
|---|---|
| split conformal, grupos finos | 0.328 |
| split conformal, grupos gruesos | 0.366 |
| pool rodante, grupos gruesos | 0.115 |
| **pool rodante + ACI (γ = 0.02), grupos gruesos** | **0.061** |
| pool rodante + ACI (γ = 0.05), grupos gruesos | 0.085 |

Tres lecturas. La primera: split conformal no se degrada, **se rompe** —marca siete veces
más de lo que declara— y lo hace sin ningún síntoma. La segunda: γ tiene el óptimo interior
que §2.2 anunciaba; 0.05 ya oscila. La tercera, y la que hay que tener presente al
configurar: la calibración condicional y la adaptación rápida compiten por la misma muestra,
así que un run que necesite reaccionar deprisa a la deriva debe usar grupos más gruesos, y
uno que necesite cobertura condicional fina debe aceptar que se adapta despacio. No es un
parámetro que se pueda dejar por defecto sin mirar: `coverage_report()` es lo que dice en
cuál de los dos regímenes se está.

### 5.3 Lo que queda fuera, y por qué

- **Día laborable / festivo, tipo de día.** El mecanismo es idéntico —una dimensión más de
  Mondrian, con `data/calendar.py` ya produciendo la columna— pero cada partición cuesta
  muestra. Se deja parametrizable y **apagado por defecto**: encenderlo debe justificarse
  midiendo cobertura condicional en `dev`, no suponiéndolo.
- **Escala aprendida `σ̂(x)`** (conformal normalizado con un modelo de varianza ajustado,
  p. ej. regresión cuantílica de `|residuo|` sobre la hora). Descartada como
  predeterminado: introduce un segundo modelo que hay que validar y que puede estar él
  mismo descalibrado, escondiendo el problema en vez de resolverlo. Mondrian es no
  paramétrico y su coste —muestra— es **visible** en `calib_n`. Es la actualización natural
  si los grupos se quedan finos, y en ese momento habrá una cifra que lo justifique.

---

## 6. Firmas públicas

```python
# chronolab/anomaly/conformal.py


@dataclass(frozen=True, slots=True)
class ConformalDetector:  # implementa AnomalyDetector
    base_model_id: ModelId | None = None
    alpha_nominal: float = 0.05  # par de cuantiles del modelo que se usa
    alpha_ref: float = 0.05  # nivel al que se mide `severity`
    alpha_grid: tuple[float, ...] = ...  # rejilla precomputada de umbrales
    gamma: float = 0.01  # 0.0 → sin ACI
    pool_size: int | None = 2000  # None → split conformal
    hour_bins: int = 6
    lead_bins: tuple[int, ...] = (1, 6, 24)
    min_calib: int = 50
    freeze_after: int = 3
    max_freeze: int | None = None  # None → 2 * spec.seasonalities[0]

    @property
    def detector_id(self) -> DetectorId: ...  # codifica gamma, pool_size y bins
    @property
    def requires(self) -> DetectorRequirements: ...
    def fit(self, calib: ScoringFrame) -> "FittedConformalDetector": ...


@dataclass(frozen=True, slots=True)
class FittedConformalDetector:  # implementa FittedDetector
    def score(self, frame: ScoringFrame) -> pd.DataFrame:
        ...
        # unique_id, ds, score, scorable, severity, calib_n, side — no muta el estado

    def advance(self, frame: ScoringFrame) -> "FittedConformalDetector": ...
    def coverage_report(self) -> pd.DataFrame:
        ...
        # level, group, n_calib, n_scored, n_flagged, flag_rate, alpha_ref,
        # alpha_eff, n_aci_clips

    def freeze_report(self) -> pd.DataFrame:
        ...
        # unique_id, n_ingested, n_frozen_steps, n_excluded, budget_exhausted
```

`DetectorRequirements(needs_forecast=True, needs_quantiles=True, window=1,
needs_calibration=True, fit_cost="cheap")`. `window=1`: el detector no consume contexto —el
calentamiento viene de la calibración, no de una ventana deslizante— así que no penaliza la
máscara `scorable` común frente a Matrix Profile o al autoencoder.

Los dos informes son la capa de honestidad. `coverage_report()` dice si la tasa marcada por
grupo se aleja de α, y con cuántos puntos se calibró cada uno. `freeze_report()` dice cuánto
se congeló la actualización en cada serie: la cuarentena de §2.4 rompe la garantía de tasa
del ACI —que asume actualización en todos los pasos— y la deja valiendo sobre la subsucesión
actualizada, así que quien lea la tasa tiene que poder ver sobre qué se cumple.
`budget_exhausted` distingue una anomalía larga de un cambio de régimen que el detector ya
se ha negado a seguir ignorando.

---

## 7. Enmiendas al contrato — aprobadas y aplicadas

| | Enmienda | Justificación |
|---|---|---|
| **A** | `ScoringFrame.df` gana `cutoff` (`timestamp[ns]`) y `h_step` (`Int16`) | Ya existen en `forecasts`; `cutoff` está desnormalizado allí justamente para evitar la unión (§7.5). Sin ellos se pierde el adelanto (escala del residuo, §5.2) y el origen de la predicción (frontera de información, §2.3), que son las dos cosas que el detector necesita estructuralmente. Es no tirar información, no crearla. Toca `anomaly/protocols.py` y §5.3 de la arquitectura. |
| **B** | `score()` y `anomaly_scores` ganan `severity` (`float32`), `calib_n` (`int32`) y `side` (`int8`) | §3.3 y §5.2. `side` no estaba en la propuesta original: la enmienda **C** pide `direction` a nivel de evento, y esa información solo existe en el punto —de qué lado del intervalo cayó la observación—, así que había que emitirla o `direction` no era calculable. |
| **C** | `anomaly_events` gana `duration_steps`, `peak_severity`, `cum_severity`, `peak_ds`, `direction` | §4. La tabla de §7.4 solo tiene `n_points` y `peak_score`, que no cubren el punto 4 del bloque. |
| **D** | `data/calendar.py` promueve `local_hour(ds, *, tz_display)` a API pública | Para que `anomaly/` no se convierta en un segundo lugar donde conviven UTC y hora local. |
| **E** | El run de detección valida `step_size == h` y `holdout_windows >= 1` | §1.3. Se valida en `artifacts.reader.scoring_frame`, es decir en el único sitio del que sale un `ScoringFrame`: un plan que no admite detección no puede producir uno. **No** se toca `BacktestPlan`. |
| **F** | §2.1 gana dos aristas: `artifacts → anomaly.protocols` y `anomaly → data.calendar` | Descubierta al implementar. La primera es una **contradicción latente del documento**: D12 exige que el lector sea el único constructor de `ScoringFrame`, y ese tipo vive en `anomaly.protocols`, así que sin la arista D12 no es implementable. La segunda es la contrapartida de **D**. Ninguna introduce ciclo: `anomaly.protocols` solo importa `panel` y `types`, y `data.calendar` no importa `anomaly`. |

`SCHEMA_VERSION` no se toca todavía: `artifacts/schemas.py` sigue siendo un stub, así que
las columnas nuevas viven de momento en la forma de los `DataFrame` que devuelven `score()`
y `aggregate_events()`, y en las tablas de §7.4 de la arquitectura, que sí se han
actualizado. La versión sube cuando se escriban los esquemas pandera.

---

## 8. Pruebas — escritas y en verde

Cuatro de ellas no comprueban código sino **afirmaciones de este documento**, y están
escritas para poder borrarse si dejan de ser ciertas: si la de heterocedasticidad falla,
§5 sobra; si la de deriva falla, §2 sobra; si la de absorción falla, §2.4 sobra.

Anti fuga (`tests/leakage/test_conformal_calibration.py`):

- `score` con `frame.start <= cutoff` lanza `CutoffViolation`; `advance` también.
- **Canario de realimentación.** Se altera la observación del último instante de una ventana
  puntuada. Esa observación solo existe *después* del origen de la ventana, así que no pudo
  entrar en ninguna de sus bandas: ningún otro punto de esa ventana puede cambiar de score.
  Es el fallo que ninguna aserción de cutoff detecta, porque lo que se usaría no es
  posterior al instante puntuado, solo al instante en que se emitió la banda.
- **La otra mitad del canario**: alterar esa observación *sí* tiene que cambiar lo que viene
  después. Sin este test, el anterior lo pasaría un detector que ignora la realimentación.
- Un plan con `step_size != h` se detiene en el lector (`tests/unit/artifacts/test_reader.py`).

Propiedades estadísticas (`tests/unit/anomaly/test_conformal.py`):

- **Cobertura.** Datos intercambiables: tasa marcada acotada por α para α en {0.01, 0.05, 0.1}.
- **Degeneración exacta.** Con `gamma=0` la inversión del nivel efectivo es la identidad, en
  toda la recta y no solo en los nodos de la rejilla.
- **Heterocedasticidad.** Con varianza dependiente de la hora, la calibración global sale con
  tasa marginal correcta (|media − α| < 0.02) y desviación condicional máxima > 0.15;
  Mondrian la deja por debajo de 0.06. Medido: 20.6 % en la punta y 1.1 % de madrugada
  frente a 4.4 % y 4.1 %.
- **Deriva.** Split conformal supera 5α; el adaptativo se queda por debajo de un tercio de
  esa tasa y de 2α. Lo que se afirma es que **recupera la mayor parte** de lo que split
  pierde, no que llegue a α exacto: bajo una rampa fuerte ningún método en línea lo hace.
- **Absorción.** Cambio de nivel inyectado de 240 pasos: sin cuarentena se marca menos del
  75 % de sus primeras 48 horas —el detector se calla solo—; con cuarentena, más del 95 %.

Unitarias (`tests/unit/anomaly/`, `tests/unit/artifacts/`):

- `score` dos veces da resultados idénticos y no avanza el estado; `advance` sí.
- Identidad `severity > 0 ⟺ score >= -log10(alpha_ref)`.
- El score satura en `log10(calib_n + 1)`, comprobado punto a punto.
- Una ventana fallida llega como `NaN`/`NaT` y sale con `scorable=False`, sin desalinear la
  rejilla.
- Fusión con `merge_gap`, `direction`, `cum_severity`, y emparejamiento uno a uno: un evento
  real partido en tres detectados da 1 acierto y 2 falsas alarmas.
- α por debajo de la resolución del score se marca `reachable=False` en vez de devolver un
  umbral que nadie podrá cruzar nunca.
- Un tramo sin la columna de cuantil no se puntúa a ciegas: falla.

---

## 9. Alternativas descartadas

| Alternativa | Motivo del descarte |
|---|---|
| Score = residuo absoluto sobre `y_hat` | Reintroduce la simetría que §7.3 declara falsa en demanda eléctrica y desperdicia los cuantiles del modelo. Queda como modo de repliegue explícito, con `detector_id` distinto, para modelos sin cuantiles. |
| Umbral global sobre residuos crudos | Cobertura marginal correcta y condicional desastrosa (§5). Es el error que el punto 5 del bloque pedía evitar. |
| Solo split conformal | Falla en silencio bajo deriva, que es el régimen normal de una serie en producción. Se conserva como línea base, no como predeterminado. |
| Solo ACI, sin pool rodante | ACI corrige el nivel pero no la forma de la distribución de no conformidad. Combinarlos es más barato que elegir. |
| Umbral variable en el tiempo en `anomaly_thresholds` | Añade `ds` a la tabla y convierte el deslizador de α en recálculo (prohibido por A5). Se pliega la adaptación dentro del score (§3.4). |
| Escala aprendida `σ̂(x)` en lugar de Mondrian | Segundo modelo que validar, capaz de estar descalibrado él mismo. Mondrian expone su coste en `calib_n` (§5.3). |
| Construir `ScoringFrame` en los tests | Abre un segundo constructor y anula la barrera de D12. |
| `Thresholder` con cuantil empírico también para conformal | El umbral es exactamente `-log10(α)` por construcción; estimarlo empíricamente añadiría error de estimación a un número que se conoce en forma cerrada. |

---

## 10. Riesgos

- **R-A1 — El modelo base emite bandas mal formadas.** Si `w_t` no tiene relación con la
  incertidumbre real, §5.1 no aporta nada y todo el peso cae en Mondrian, que puede no tener
  muestra. Detección: `coverage_report()` con tasas por grupo lejos de α pese al repliegue.
- **R-A2 — Presupuesto muestral insuficiente para α pequeña.** Es el límite duro de §5.2. Se
  gestiona truncando la rejilla y registrándolo, no extrapolando.
- **R-A3 — γ ajustado sobre holdout.** Anularía la disciplina de `stage`. Mitigación:
  el punto de entrada del run recibe solo ventanas `dev` para tuning, igual que el
  optimizador de forecasting.
- **R-A4 — La verdad de anomalías es sintética** (R4 de la arquitectura). Todas las métricas
  de detección son relativas a la inyección de `anomaly/injection.py`; no hay incidentes
  etiquetados reales. Este diseño no cambia ese riesgo y no debe presentarse como si lo
  hiciera.
