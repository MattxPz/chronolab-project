# Auditoría externa de chronolab

> **Rol:** revisor externo escéptico, contexto de decisión de contratación.
> **Fecha:** 2026-08-09. **Commit auditado:** `8bcb5d8`.
> **Alcance:** camino completo del dato (ingesta → panel → ventanas → features →
> modelo → métrica → artefacto → documento), evaluación de anomalías,
> reproducibilidad y contraste entre lo que la documentación afirma y lo que el
> código hace.
> **Este documento solo diagnostica. No se ha modificado nada más.**

---

## Veredicto en una página

El marco de evaluación es, en su diseño, mejor que la mayoría de lo que se ve en
repositorios de forecasting: el denominador de MASE se calcula por serie y por
ventana con el train de esa ventana, el splitter es el único emisor de
particiones, `FutrFrame` impide leer exógenas históricas por ausencia física, y
el módulo de métricas de anomalía rechaza el *point-adjusted F1* con un test
adversario que reproduce su inflación. Eso hay que reconocerlo antes de nada, y
está detallado en [§5](#5-lo-que-sí-aguanta-el-examen).

El problema es que **ese diseño no es el que produjo los números publicados**.
Los cuatro hallazgos críticos no son fallos de concepto: son la distancia entre
las barreras que el código describe y las que efectivamente se ejecutaron al
generar `reports/results/` y los documentos de `docs/`.

| Severidad | Nº | Efecto |
|---|--:|---|
| Crítico | 4 | Invalidan los resultados publicados tal y como están |
| Grave | 12 | Debilitan las conclusiones o hacen la comparación desigual |
| Menor | 11 | Calidad, consistencia, mantenimiento |

Los cuatro críticos, resumidos:

1. **El invariante sobre el que descansa todo el aparato anti-fuga no está
   implementado.** `Panel` no valida nada y `assemble.build_panel` —descrito
   como "el único constructor público"— es un fichero de cuatro líneas con un
   docstring. Demostrado abajo: un panel con un hueco cambia el denominador de
   MASE de `0.0` a `0.2817` y se construye sin error.
2. **El umbral común de anomalías no está calibrado**, y el documento de
   hallazgos afirma explícitamente que sí. A `alpha = 0.05`, IsolationForest
   marca el **48.6 %** de los instantes y el LSTM-AE el **53.5 %**. Su
   `range_recall` de 0.90 no es detección: es cobertura por saturación.
3. **El leaderboard publicado se generó con presciencia perfecta**, el aviso que
   existe para impedirlo se silencia a mano en los dos scripts que lo producen,
   y el artefacto no lleva ninguna marca de *vintage*. Además es asimétrico:
   solo 4 de los 19 modelos consumen la exógena futura — y **los dos primeros
   puestos de la tabla están entre esos cuatro**.
4. **El hallazgo de portada del README no es reproducible.** Ningún artefacto lo
   respalda, sus cifras contradicen a `leaderboard.parquet`, y el modelo del que
   trata (`chronos`) no aparece en ninguna fila de ningún artefacto del repo.

---

## Cómo se ha auditado

Lectura completa de `src/chronolab/evaluation/`, `features/`, `anomaly/`,
`panel.py`, los seis adaptadores de modelo, los cuatro scripts de `scripts/` y
los documentos de `docs/`. Ejecución de:

- `uv run pytest` → **1274 pasados en 94 s**, sin fallos.
- Inspección directa de los `.parquet` de `reports/results/`.
- Dos contraejemplos ejecutables (C1 y §4), reproducibles con los comandos del
  [apéndice](#apéndice-comandos-de-reproducción).

Nada de lo que sigue es una sospecha de lectura: todo está o verificado sobre el
artefacto, o demostrado con código ejecutado.

---

## 1. Crítico — invalida resultados

### C1. `Panel` no valida nada; los invariantes I1–I7 son prosa

**Dónde.** [`src/chronolab/panel.py:116-141`](../src/chronolab/panel.py#L116-L141)
(la clase `Panel` no tiene `__post_init__`),
[`src/chronolab/data/assemble.py`](../src/chronolab/data/assemble.py) (4 líneas,
solo docstring).

El docstring de `Panel` afirma:

> *"Los invariantes I1-I7 se garantizan en construcción. Ningún consumidor debe
> volver a comprobarlos y ningún productor puede saltárselos, porque el único
> constructor público es `chronolab.data.assemble.build_panel`."*

`build_panel` **no existe**. `grep -rn "build_panel" src/ tests/ scripts/`
devuelve tres referencias, todas en comentarios y docstrings —una de ellas,
`scripts/refresh_data.py:277`, admitiendo "sin implementar"—. `Panel` es un
`dataclass` congelado sin ninguna validación, y todo el proyecto lo construye
directamente.

**Por qué importa.** Tres piezas centrales presuponen I3 (rejilla completa) e I4
(orden por `(unique_id, ds)`) y ninguna vuelve a comprobarlo, porque el docstring
les dice que no hace falta:

- `RollingOriginSplitter.split` hace aritmética sobre `panel.grid()`, que es
  `date_range(first_ds, last_ds, freq)` —no las filas reales—. Con huecos, las
  ventanas se anclan a instantes que no existen.
- `features.ops.lag/roll/diff` desplazan **posicionalmente**
  (`groupby.shift(k)`): con una fila ausente, `lag(y, 24)` no es el valor de hace
  24 horas.
- `seasonal_naive_mae` compara `values[season:]` con `values[:-season]` sobre el
  array crudo.

**Demostración.** Serie horaria perfectamente periódica, estacionalidad 24, a la
que se le quita **una** fila:

```
q con rejilla completa      : 0.0
q con una fila desaparecida : 0.28169
Panel(gapped) aceptado sin error. len(df)= 95  len(grid())= 96
```

El denominador de MASE —la escala de la métrica principal del proyecto— cambia
de indefinido (correcto: el naive estacional no comete error) a `0.2817`, y el
`Panel` se construye sin una sola advertencia. También acepta filas
desordenadas.

**Cómo se arregla.** Implementar `assemble.build_panel` con las validaciones I1–I7
usando los esquemas pandera que ya existen en `data/schemas.py`, y añadir un
`__post_init__` barato a `Panel` que verifique al menos I3 e I4
(`len(df) == len(grid()) * n_series` y monotonía por grupo) o, si el coste
preocupa, un flag `validated: bool` que `build_panel` sea el único en poder
poner. Mientras tanto, el docstring de `Panel` debe decir la verdad.

---

### C2. El umbral común de anomalías no está calibrado, y el documento afirma que sí

**Dónde.** [`docs/ANOMALY_FINDINGS.md` §1](ANOMALY_FINDINGS.md),
[`scripts/run_anomaly_eval.py:99-100`](../scripts/run_anomaly_eval.py#L99-L100),
artefacto `reports/results/anomaly_scores.parquet`.

`ANOMALY_FINDINGS.md` afirma:

> *"**El umbral común es legítimo aquí y no siempre lo sería.** Los cuatro
> detectores emiten `-log10(p)` con `p` un p-valor conformal, así que
> `alpha = 0.05` significa lo mismo en los cuatro."*

Tasa de marcado real, medida sobre el propio artefacto, dentro de la máscara
`scorable` común y con el umbral que el documento declara:

| Detector | Marcado a α=0.05 | Esperado |
|---|--:|--:|
| Conformal (CQR) | 8.4 % | ~5 % |
| MatrixProfile | 4.8 % | ~5 % (pero ver G2) |
| **IsolationForest** | **48.6 %** | ~5 % |
| **LSTM-Autoencoder** | **53.5 %** | ~5 % |

Dos detectores están descalibrados **en un orden de magnitud**. `alpha = 0.05`
no significa lo mismo en los cuatro; significa cosas distintas por un factor
diez.

**Por qué importa.** Es la conclusión de portada del hito de anomalías. La tabla
de `ANOMALY_FINDINGS.md` §2 presenta como hallazgo que *"el ganador depende de la
métrica"*, con LSTM-AE (0.900) e IsoForest (0.893) encabezando `range_recall` y
Conformal (0.434) muy por detrás. Ese orden es consecuencia mecánica del umbral
roto: **quien marca la mitad de la línea temporal toca casi todos los eventos**.
Sus `range_precision` de 0.226 y 0.140 lo confirman —el 86 % de sus marcas son
falsas alarmas—. La discrepancia entre familias de métricas, que el documento
usa para justificar todo el módulo, está en buena parte fabricada por un
artefacto de calibración, no por una tensión real entre criterios.

La causa mecánica más probable —no verificada, y por eso se anota como
hipótesis— es que ambos detectores son de ventana (`window=24`, `seq_len=24`) y
cada anomalía inyectada contamina las features de los 24 pasos siguientes: 54
eventos × 24 ≈ 1300 de 2979 instantes, que es del orden del 44 % observado.

**Cómo se arregla.** Antes de publicar: medir la tasa de marcado de cada detector
sobre un tramo **limpio** de holdout y comprobar que se acerca a α. Si no lo
hace, el detector no está calibrado y comparar a α fijo es ilegítimo —hay que
comparar a tasa de marcado fija, o publicar la curva completa—. La corrección
mínima honesta es añadir la columna "tasa de marcado observada" a las tablas de
`ANOMALY_FINDINGS.md` y retirar la afirmación de legitimidad del umbral común.

---

### C3. El leaderboard se generó con presciencia perfecta, silenciada a mano y sin etiquetar

**Dónde.**
[`scripts/run_ml_feature_analysis.py:469-470`](../scripts/run_ml_feature_analysis.py#L469-L470),
[`scripts/run_deep_analysis.py:465-466`](../scripts/run_deep_analysis.py#L465-L466):

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PerfectForesightWarning)
    futr = RealizedFutrProvider(panel=panel)
```

`RealizedFutrProvider` está documentado sin ambigüedad en
[`data/futr.py:62-97`](../src/chronolab/data/futr.py#L62-L97):

> *"El resultado de un run con este proveedor es una **cota superior** de
> rendimiento, no una estimación de lo que el sistema lograría en producción, y
> solo es publicable si se etiqueta como tal. Por eso el aviso se emite al
> construirlo."*

El aviso se emite. Los dos scripts que producen `leaderboard.parquet` lo apagan.
Ni el artefacto, ni `DEEP_ANALYSIS.md`, ni `FEATURE_ANALYSIS.md`, ni el README
llevan la etiqueta de cota superior.

**Por qué importa, y por qué es peor que un simple optimismo.** El sesgo es
**asimétrico**, y cae justo del lado de los ganadores. El panel declara `temp_c`
como `futr_exog`, y solo la consumen los modelos que declaran
`needs_futr_exog`:

| Reciben la temperatura futura **real** (4) | No la reciben (15) |
|---|---|
| `prophet` — `regressors = ("temp_c",)` por defecto (`prophet.py:88`), y los scripts lo instancian sin argumentos | `patchtst` — `use_futr_exog=False` explícito en el script |
| `nhits`, `tft` — `use_futr_exog: bool = True` por defecto (`neuralforecast.py:659`) | `lightgbm_*`, `xgboost_*` — el adaptador de mlforecast **nunca lee el `FutrFrame`**, por diseño documentado |
| `lstm` — `use_futr_exog: bool = True` por defecto (`torch_lstm.py:298`) | `auto_arima`, `auto_ets`, `auto_theta`, `mstl` y los seis baselines |

**Los dos primeros puestos del leaderboard publicado — `prophet` (MASE 0.2794) y
`nhits` (0.2843) — están ambos en el grupo privilegiado.** Quince modelos
compiten a ciegas contra cuatro que conocen la temperatura exacta del futuro, y
los diecinueve se ordenan por MASE en la misma tabla, presentada como comparación
limpia sobre "exactamente las mismas ventanas". `DEEP_ANALYSIS.md` concluye de
ahí que *"Mejor modelo del leaderboard: **prophet**"*: es la conclusión que este
sesgo produce, no una que sobreviva a quitarlo.

En un panel donde la demanda se genera con dependencia térmica explícita, la
ventaja de conocer `temp_c` a 24 horas no es marginal — es plausiblemente la
mayor parte de la diferencia entre el primer puesto y el cuarto.

Un agravante estructural: `BacktestResult.futr_vintage` guarda `Vintage.REALIZED`
en memoria, pero `build_leaderboard` no lo escribe en ninguna columna, así que la
información se pierde al persistir. `aggregate.py` abre su docstring con *"no se
comparan filas con distinto `futr_vintage`"* y el artefacto no contiene el campo
que haría posible comprobarlo.

**Cómo se arregla.** (a) Propagar `futr_vintage` a una columna del leaderboard y
rechazar mezclar vintages. (b) No silenciar `PerfectForesightWarning`: dejarlo
salir por consola y estamparlo en el documento generado. (c) Para que la
comparación sea legítima, o bien correr con `SimulatedForecastProvider` (un
pronóstico degradado con error realista), o bien correr el leaderboard sin
exógenas futuras y publicar el run con presciencia como *ablación* etiquetada.

---

### C4. El hallazgo de portada del README no es reproducible ni consistente

**Dónde.** [`README.md:35-85`](../README.md) frente a
`reports/results/leaderboard.parquet` y `docs/DEEP_ANALYSIS.md`.

El README dedica su única sección de resultados a *"zero-shot frente a modelos
ajustados"*, con esta tabla:

| README | MASE | `leaderboard.parquet` | MASE |
|---|--:|---|--:|
| MSTL | **0.296** | `mstl` | **0.2877** |
| Chronos-Bolt-small | **0.313** | — *(ausente)* | — |
| Naive estacional (24 h) | **0.902** | `seasonal_naive` | **0.7269** |
| AutoETS | **1.066** | `auto_ets` | **0.6533** |

- **`chronos` no aparece en ningún artefacto del repositorio.** El leaderboard
  tiene 19 modelos y `is_zero_shot = False` en las 19 filas. El script que lo
  genera (`run_deep_analysis.py`) no instancia el adaptador de Chronos.
- Ninguna de las cuatro cifras coincide con el artefacto versionado. AutoETS
  difiere en un 63 %.
- El README dice "6 ventanas, **3** de holdout"; los dos scripts usan
  `HOLDOUT_WINDOWS = 2`, y el artefacto lo confirma (`n_windows = 2`).
- La columna se titula "`fit_seconds` (total, 6 ventanas)", pero
  `_with_timings` filtra `model_runs` por etapa: con `stage="holdout"` el total
  cubre 2 ventanas, no 6.

**Por qué importa.** Es el primer número que lee cualquiera que abra el
repositorio, es el que la sección presenta como su aportación, y no hay forma de
regenerarlo ni de auditarlo. En un contexto de contratación, un resultado de
portada que no se puede reproducir con el código del propio repositorio pesa más
en contra que la ausencia del resultado.

**Cómo se arregla.** O se añade Chronos al run que produce `leaderboard.parquet`
y se regenera la tabla del README desde el artefacto (como ya hace
`DEEP_ANALYSIS.md`, que sí es trazable), o se retira la sección. La mezcla actual
—una tabla escrita a mano que contradice al artefacto— es la peor de las tres
opciones.

---

## 2. Grave — debilita las conclusiones

### G1. Todo el leaderboard descansa en 2 ventanas y 144 observaciones por modelo

`n_windows = 2`, `n_obs = 144` (3 series × 24 pasos × 2 ventanas) en las 19
filas agregadas. Las tres primeras posiciones son `prophet` 0.2794, `nhits`
0.2843, `mstl` 0.2877: separaciones del 1–3 % sobre 144 puntos, sin intervalo de
confianza ni test. `evaluation/stats_tests.py` implementa Diebold-Mariano con
corrección HLN y ajuste de p-valores múltiples —y **no se aplica al
leaderboard**; solo se usa en `build_demo_artifacts.py` sobre otro run distinto.
Publicar un orden de 19 modelos sin contraste sobre esa muestra no es un
ranking, es ruido ordenado.
*Arreglo:* subir `n_windows`, y emitir la matriz DM (o un bootstrap por bloques
sobre ventanas) junto al leaderboard, no en un artefacto aparte.

### G2. MatrixProfile es transductivo y se autocalibra sobre el tramo que puntúa

[`anomaly/matrix_profile.py:229-252`](../src/chronolab/anomaly/matrix_profile.py#L229-L252).
Dos problemas distintos, de los cuales el docstring del módulo reconoce solo el
segundo:

1. `stumpy.stump(values, m)` se calcula sobre **todo el tramo de holdout de una
   vez**. El vecino más cercano de una subsecuencia en `t` puede estar en
   `t + k`. El score en `t` depende de datos futuros. El módulo lo llama
   "puntuación retrospectiva" y lo justifica porque audita un tramo cerrado,
   pero los otros tres detectores no tienen esa libertad y compiten en la misma
   tabla.
2. `_self_referential_score(pool, pool)` usa como pool de referencia **las
   mismas distancias que está puntuando**. Esto tiene una consecuencia
   mecánica que ni el docstring ni el documento de hallazgos mencionan: con
   `p = (1 + count_ge) / (n + 1)`, la fracción de puntos que cruzan
   `-log10(α)` queda **fijada en α por construcción**, independientemente de
   cuántas anomalías haya. El 4.83 % medido no es evidencia de calibración: es
   una identidad algebraica.

Su `auc_pr` es 0.175 — exactamente la prevalencia, es decir, el azar. Presentarlo
como "referencia de contraste del arnés" cuando está al nivel del azar y además
ve el futuro no informa de nada.
*Arreglo:* calcular el perfil en modo incremental (`stumpy.stumpi`) o al menos
restringir el vecino a `índice < t`, y tomar el pool de p-valores de un tramo de
calibración disjunto. Si se prefiere dejarlo como está, sacarlo de las tablas
comparativas y publicarlo como "cota superior sin calibración".

### G3. Un tipo de anomalía de seis nunca se evaluó

`anomaly_results.parquet` da `n_events = 0` para `data_gap` en los **cuatro**
detectores. La causa: `data_gap` pone `y = NaN`, esos instantes dejan de ser
`scorable`, y desaparecen de la máscara común. `ANOMALY_FINDINGS.md` anuncia
*"4 detectores × 6 tipos de anomalía"* y *"54 eventos"*, y luego sus tablas
dicen `n_events = 45` sin explicar los 9 que faltan. El mapa de calor dibuja la
columna con "n/d", lo cual es correcto y honesto en la figura, pero el texto no
lo recoge.
*Arreglo:* decir en el documento que `data_gap` no es evaluable con esta máquina
(un instante ausente no puede puntuarse) y bajar el titular a 5 tipos, o
rediseñar el tipo para que el hueco se manifieste en una feature puntuable.

### G4. Tuning asimétrico: dos modelos afinados, diecisiete no

`scripts/run_ml_feature_analysis.py:76` fija `N_TRIALS = 8` y solo tunea
LightGBM y XGBoost. Los hiperparámetros resultantes se hardcodean en
[`run_deep_analysis.py:96-106`](../scripts/run_deep_analysis.py#L96-L106) y se
reutilizan. Los otros diecisiete modelos —incluidos los tres profundos y el
LSTM propio— corren con valores por defecto o presupuestos elegidos a ojo
(`max_steps=150`, `windows_batch_size=64`). La conclusión de `DEEP_ANALYSIS.md`
—*"Mejor modelo profundo: nhits"*, *"el LSTM propio pierde"*— compara modelos
con presupuestos de búsqueda incomparables.
El tuning en sí es correcto (usa `dev_only_panel`, no ve holdout). El problema
es la asimetría, y que ninguna tabla publicada la menciona.
*Arreglo:* o se tunean todos con el mismo presupuesto, o cada fila del
leaderboard lleva una columna `n_trials` con el suyo (cero incluido).

### G5. Política de refit asimétrica en el run que produjo el leaderboard

`run_ml_feature_analysis.py:460-466` deja `refit_every` sin fijar a propósito, y
`BacktestPlan.refit_every_for` aplica entonces: 1 ajuste por ventana para
`cheap`/`free`, **un solo ajuste por run** para `expensive`. Es decir: los cuatro
modelos de statsforecast se ajustan una única vez, en la ventana más corta, y
reutilizan ese ajuste; LightGBM se reajusta en las seis. El comentario del script
lo dice; el leaderboard no. `model_runs.refit_every` lo registra, pero
`build_leaderboard` no lo propaga a la tabla publicada. Un MASE de AutoARIMA
obtenido con un ajuste de hace cinco ventanas y uno de LightGBM reajustado en
cada una no son la misma cantidad.
*Arreglo:* propagar `refit_every` y `n_refits` (esta ya está) a la tabla, y
correr el leaderboard de referencia con `refit_every=1` para todos, como sí hace
`run_deep_analysis.py`.

### G6. `tune()` recibe un proveedor atado al panel completo — justo lo que su docstring prohíbe

[`tuning.py:248-254`](../src/chronolab/evaluation/tuning.py#L248-L254) advierte:

> *"Si está construido sobre un `Panel`, debe construirse sobre el panel
> recortado que devuelve `dev_only_panel(panel, plan)`, no sobre `panel`: pasar
> aquí un proveedor atado al panel completo reintroduce por otra puerta la fuga
> que este módulo existe para impedir."*

`run_ml_feature_analysis.py:470-474` construye `RealizedFutrProvider(panel=panel)`
sobre el panel **completo** y lo pasa a `_tune_one(panel, plan, futr, ...)`.

**Matiz que hay que dar, porque el revisor no debe exagerar:** en esta
configuración concreta **no hay fuga efectiva**. `dev_only_panel` recorta hasta
`dev_end`, el plan reescalado se ancla en ese nuevo final, y todas las ventanas
`dev` piden exógenas dentro de `[…, dev_end]` —filas idénticas en el panel
completo y en el recortado—. La fuga es latente, no consumada.

Pero nada la impide. Basta un `gap > 0`, un `step_size` distinto o un cambio en
el anclaje del splitter para que empiece a serlo, y no hay test ni aserción que
lo detecte. Un contrato documentado que el propio repositorio incumple en su
único uso real deja de ser un contrato.
*Arreglo:* que `tune()` construya el proveedor internamente sobre
`trimmed_panel`, o que compruebe `futr.panel.last_ds <= trimmed_panel.last_ds` y
falle si no. Añadir el test correspondiente a `tests/leakage/`.

### G7. Prevalencia de anomalías del 17.5 %

522 instantes anómalos de 2979. La detección de anomalías en producción trabaja
entre el 0.1 % y el 1 %. A esta prevalencia, la línea base de AUC-PR es 0.175
—casi cuatro veces la de un problema realista—, la precisión es
estructuralmente fácil, y el pool de referencia de cualquier detector que
calibre sobre el propio tramo (G2) queda contaminado. Se agrava con el desajuste
de granularidad: los detectores de ventana 24 marcan necesariamente una
vecindad de 24 pasos alrededor de cada evento, mientras la verdad etiqueta solo
los instantes inyectados, así que su `range_precision` está penalizada por
construcción y su tasa de marcado inflada por construcción.
*Arreglo:* bajar la densidad de inyección a un régimen realista (o publicar
varias prevalencias y mostrar la sensibilidad), y añadir una tolerancia de
vecindad a la verdad al evaluar detectores de ventana.

### G8. `build_leaderboard` no impone que los modelos se comparen sobre las mismas filas

[`aggregate.py:219-237`](../src/chronolab/evaluation/aggregate.py#L219-L237)
agrupa por `(model_id, unique_id)` y puntúa lo que haya. Un modelo que falla en
una ventana simplemente no aporta filas, y su MASE se calcula sobre un
subconjunto distinto del de sus rivales. Igualmente, `point_metrics` descarta
pares con `NaN`, así que un modelo que emite `NaN` en algunas predicciones se
evalúa sobre menos puntos. Las columnas `n_obs`, `n_windows` y
`n_windows_failed` **exponen** el problema —eso es más de lo que hace casi
nadie— pero no lo **impiden**, y el orden de la tabla no lo tiene en cuenta.
En el run publicado todos los modelos tienen `n_windows_ok = 2, failed = 0`, así
que aquí no muerde; el riesgo es estructural.
*Arreglo:* calcular la intersección de claves `(unique_id, window_id, ds)` con
`y_hat` finito sobre todos los modelos del run —el equivalente de
`common_scorable_mask`, que el módulo de anomalías ya hace bien— y puntuar sobre
ella, emitiendo aparte las métricas sobre soporte propio.

### G9. CI no ejecuta ni una línea del núcleo de modelado

`.github/workflows/ci.yml`:

- Instala `uv sync --extra api`: **sin `ml` ni `deep`**. Los seis adaptadores de
  modelo, los cuatro detectores y el tuning se saltan por `importorskip`.
- El comentario de la línea 39 dice que esos extras *"se ejercitan en el job
  `smoke`"*. **Ese job no existe** en el fichero: solo hay `quality` y
  `artifact-size`.
- `env: UV_FROZEN: "0"` desactiva el lockfile. Con todas las dependencias
  declaradas como `>=` y sin techo, dos ejecuciones separadas en el tiempo
  resuelven a versiones distintas.

Es decir: la señal verde de CI cubre lint, mypy y los tests que no necesitan
modelos. Localmente sí pasa la suite completa (1274 tests, verificado), pero eso
depende del entorno de una máquina concreta.
*Arreglo:* añadir el job `smoke` que el comentario promete, con
`uv sync --all-extras` y los tests marcados `slow`; poner `UV_FROZEN: "1"`;
acotar por arriba al menos las dependencias que producen números (`numpy`,
`pandas`, `statsforecast`, `mlforecast`, `torch`).

### G10. La cobertura de los intervalos no se contrasta contra la nominal en ninguna parte

El leaderboard emite `coverage_50`, `coverage_80` y `coverage_95`, y nada compara
esos números con 0.50, 0.80 y 0.95. Los valores publicados:

| Modelo | `coverage_50` | `coverage_95` |
|---|--:|--:|
| `window_average` | 0.368 | 1.000 |
| `naive` / `drift` | 0.375 | 0.951 |
| `historic_average` | 0.403 | 1.000 |
| `lstm` | 0.451 | 1.000 |
| `prophet` | 0.486 | 0.958 |
| `nhits` | 0.618 | 0.986 |
| `patchtst` | 0.611 | 0.993 |

Los intervalos al 50 % cubren entre el 37 % y el 62 %. Los del 95 % llegan al
100 %. Ninguno de esos desvíos dispara nada: no hay aviso, no hay columna
"desviación de la nominal", no hay test de calibración. El docstring de
`empirical_coverage` explica correctamente que hay que compararla con la nominal
—y luego nadie lo hace—.
*Arreglo:* emitir `coverage_error_<nivel> = empírica − nominal` en el leaderboard
y un intervalo binomial exacto sobre `n_obs_prob`, para que se vea qué desvíos
son señal y cuáles son los 144 puntos de muestra.

### G11. Ocho de diecinueve modelos no tienen ninguna métrica probabilística, y compiten igual

`n_obs_prob = 0` en `mstl`, `lightgbm_direct`, `lightgbm_recursive`,
`xgboost_recursive`, `xgboost_direct`, `auto_theta`, `auto_ets`, `auto_arima`.
Causa real: los scripts los construyen con `use_intervals=False`
(`run_deep_analysis.py:117-120`). El leaderboard se ordena por MASE, así que no
producir intervalos no cuesta nada: `mstl` es tercero global sin haber emitido
una sola banda. `ConformalWrapper`, el módulo que existe precisamente para
convertir cualquier `Forecaster` puntual en probabilístico, es un stub (ver G12).
*Arreglo:* implementar `ConformalWrapper` y envolver a los ocho, o marcar la
tabla con una columna `probabilistic: bool` y no mezclar los dos regímenes en un
único orden.

### G12. Cuatro módulos documentados como infraestructura central son stubs vacíos

| Módulo | Líneas | Lo que la arquitectura dice que hace |
|---|--:|---|
| `data/assemble.py` | 4 | Constructor único y validador de `Panel` (→ C1) |
| `models/wrappers.py` | 5 | `ConformalWrapper` (→ G11) |
| `models/registry.py` | 6 | Registro `model_id` → fábrica desde `conf/models.yaml` |
| `artifacts/writer.py` | 6 | Escritura atómica del run, `manifest.json`, `run_id` |

Los cuatro contienen solo un docstring en presente de indicativo describiendo un
comportamiento inexistente. `tests/unit/test_module_tree.py` los da por buenos
porque solo comprueba que el módulo **importe**, cosa que un fichero con un
docstring hace perfectamente. Consecuencias encadenadas: no hay `run_id`
persistido, no existe `metrics.parquet` ni la tabla `runs` de la §7 de la
arquitectura, y por tanto `futr_vintage` no llega a disco (→ C3) y ningún
resultado es rastreable a una configuración.
*Arreglo:* implementarlos o marcarlos explícitamente como `NotImplementedError`
con un `TODO` fechado, y endurecer `test_module_tree.py` para que compruebe la
presencia de los símbolos públicos que la arquitectura declara, no solo el
import.

---

## 3. Menor — calidad y consistencia

| # | Hallazgo | Dónde |
|---|---|---|
| M1 | El README dice *"Estado: andamiaje. […] los algoritmos todavía no"* sobre 38 000 líneas con 19 modelos, 4 detectores, app y API. Se infravalora el trabajo hecho, y contradice al resto de `docs/`. | `README.md:5-8` |
| M2 | Enlace roto: el README apunta a `docs/PLAN_PROYECTO.md`; el fichero está en la raíz. | `README.md:8` |
| M3 | El README atribuye la ausencia de métricas probabilísticas a *"parece un efecto de `ConformalIntervals` con `refit_every=1`"* y añade *"no se ha investigado la causa"*. La causa está escrita en el propio script: `use_intervals=False`. Especular sobre el comportamiento del código propio cuando la respuesta está a un `grep` resta credibilidad al resto del documento. | `README.md:81-85` |
| M4 | `leaderboard.parquet` no tiene la columna `training_regime` que `_TIMING_COLUMNS` declara: el artefacto versionado se generó con una versión anterior de `aggregate.py` y nunca se regeneró. | `aggregate.py:53-65` |
| M5 | `BacktestPlan.seed` está declarado, documentado como "semilla global del run" y **nunca se lee** en ninguna parte del motor. Sugiere un control de reproducibilidad que no se ejerce. | `backtest.py:150` |
| M6 | Al restringir por tipo, `evaluate_detector` elimina posiciones del array y comprime lo que queda. El comentario afirma que no afecta *"porque lo que se emite en este grano no depende de la adyacencia entre instantes lejanos"*, pero `range_precision_recall` llama a `runs_to_ranges(..., merge_gap)`, que sí depende de la adyacencia: dos marcas antes separadas pueden quedar contiguas y fundirse en un solo rango. | `anomaly_metrics.py:1350-1377` |
| M7 | `variance_shift` etiqueta como anómalos los 171 instantes de su tramo aunque el desplazamiento sea ≈0 allí donde `current ≈ local_mean`; `sensor_freeze` congela en `current[0]`, dejando su primer punto prácticamente intacto y aun así etiquetado. La verdad de referencia contiene puntos indetectables por construcción, que deprimen el recall de los cuatro detectores por igual. | `anomaly/injection.py:371-382` |
| M8 | Ningún número publicado procede de datos reales: todo viene de `tests.fixtures.synthetic.make_hourly_panel`. `REEDemandSource`, `UCIElectricitySource` y `BinanceSource` están implementados y testeados pero no alimentan ningún resultado; solo `OpenMeteoSource` y REE se usan, y únicamente en `refresh_data.py`, cuya salida está gitignorada. El README lo reconoce para Chronos ("panel sintético, no productivo") pero no para el leaderboard entero. | `scripts/*.py` |
| M9 | Los scripts de `scripts/` importan de `tests/` (`from tests.fixtures.synthetic import make_hourly_panel`) tras un `sys.path.insert`. Un generador de datos del que dependen los artefactos publicados debería vivir en `src/` —donde ya existe `data/sources/synthetic.py`, que no se usa para esto—. | `run_anomaly_eval.py:58`, `run_deep_analysis.py:42`, `run_ml_feature_analysis.py:42` |
| M10 | `quality.detect_outliers` usa mediana y MAD **globales** de la serie completa: no es causal. No realimenta al modelado (solo produce el artefacto de calidad), así que no hay fuga, pero se publica en la app junto a un pipeline que presume causalidad estricta y conviene que la página lo diga. | `data/quality.py:199-208` |
| M11 | `config._REPO_ROOT = Path(__file__).resolve().parents[2]` resuelve a `site-packages/..` si el paquete se instala como wheel. Funciona solo ejecutando desde el repositorio. | `config.py:23` |

---

## 4. Riesgo latente verificado por lectura, no ejercitado

`_FittedMLForecastModel._future_regressors` reconstruye las térmicas
concatenando la historia de entrenamiento con la rejilla futura y recalculando
`ops.lag`, que desplaza **posicionalmente**. Con `gap > 0` esa concatenación deja
un agujero de `gap` pasos entre el final del train y `first_pred`, así que
`lag(temp, 24)` dejaría de significar "hace 24 horas". El motor lo detectaría
—`_validate_prediction` compara la rejilla emitida con la evaluada— y la ventana
saldría como `status="failed"`, no como un número silenciosamente equivocado.
Ningún run del repositorio usa `gap > 0`, así que no está ejercitado.
Merece un test en `tests/leakage/` que fije el comportamiento.

---

## 5. Lo que sí aguanta el examen

Un informe que solo enumera defectos no es una auditoría, es una queja. Lo
siguiente se ha revisado buscando fallos y no se han encontrado:

- **El denominador de MASE.** `mase_denominators` recorta el panel a
  `[train_start, cutoff]` de cada ventana y calcula `seasonal_naive_mae` serie a
  serie. Se persiste, es auditable, y el valor es idéntico en las 19 filas del
  leaderboard —comprobado— porque las ventanas y series son las mismas. La
  agregación escala fila a fila antes de promediar, no promedia y luego divide.
  Está bien.
- **`RollingOriginSplitter`.** Aritmética sobre la rejilla, anclada al final,
  `first_pred = cutoff + (gap+1)·freq` verificado en `__post_init__`, y el
  `stage` decidido sobre la numeración del plan y no sobre las supervivientes.
  No hay forma de pedir un split aleatorio: la función no existe.
- **`features.ops`.** `roll`/`expand`/`ewm` exigen `shift >= 1` sin valor por
  defecto que alguien pueda dejarse, no hay parámetro `center`, y `lead` falla
  al construirse sobre una columna de `max_lead` finito. El álgebra de
  `max_lead` se calcula, nunca se declara. Es la barrera más elegante del
  repositorio.
- **Las barreras del motor de backtesting.** `_assert_train_at_cutoff`,
  `_assert_futr_frame`, `_assert_prediction_after_cutoff` y
  `_assert_fitted_at_or_before_cutoff` están en el único camino que produce
  predicciones, y `LeakageError` se re-lanza explícitamente en vez de caer en el
  `except Exception` que registra fallos de modelo. Correcto.
- **`FutrFrame`.** Ausencia física de las columnas prohibidas, no ausencia por
  convenio. Es la forma fuerte.
- **El rechazo del *point-adjusted F1*.** No solo no se implementa: hay un test
  adversario (`test_anomaly_metrics.py:308-320`) que reproduce la inflación
  —ruido aleatorio con F1-PA > 0.55 y > 5× el honesto— para dejar constancia de
  por qué. Y `CardinalityMode="reciprocal"` cierra la puerta trasera de la
  fragmentación.
- **`common_scorable_mask`.** Intersecta las máscaras antes de comparar
  detectores. Es exactamente lo que le falta al leaderboard de forecasting (G8).
- **`dev_only_panel`.** La barrera contra el tuning sobre holdout es ausencia
  física de datos, no una comprobación en tiempo de ejecución. El diseño es
  correcto; el problema de G6 está en cómo lo llama el script, no en el módulo.
- **`_repair_quantile_crossing`.** Ordena solo los valores finitos de cada fila,
  cuenta las reparaciones y las publica. La alternativa obvia (`np.sort` sobre
  la fila entera) habría pisado cuantiles reales con `NaN`.
- **La suite de tests.** 1274 pasan en 94 s. `tests/leakage/` contiene los seis
  tests que a un revisor le importan (canario de exógenas, estabilidad por
  prefijos, disjunción de ventanas, denominador de MASE, escalado solo con
  train, calibración conformal).

---

## 6. Orden de reparación sugerido

No es una lista de deseos: es lo que hay que hacer, y en qué orden, para que los
números publicados vuelvan a ser defendibles.

1. **C1** — implementar `build_panel` y la validación de `Panel`. Es la base de
   todo lo demás y es media jornada.
2. **C3 + G5 + G11** — regenerar el leaderboard en un único run: sin presciencia
   perfecta (o etiquetada), `refit_every=1` para todos, intervalos activos para
   todos. Un solo run limpio invalida y sustituye a los tres actuales.
3. **C2 + G2 + G3 + G7** — rehacer el experimento de anomalías: medir la tasa de
   marcado en tramo limpio antes de comparar, bajar la prevalencia, resolver o
   retirar `data_gap`, y sacar MatrixProfile de la comparación directa.
4. **C4 + M1–M4** — reescribir el README desde los artefactos.
5. **G9** — arreglar CI: job `smoke`, `UV_FROZEN=1`, techos de versión.
6. **G1 + G10** — añadir contraste estadístico y error de cobertura al
   leaderboard. Con más ventanas del punto 2, esto ya es barato.
7. El resto.

---

## Apéndice: comandos de reproducción

```bash
# Suite completa (verificado: 1274 pasados, 94 s)
uv run pytest -q

# C1 — el Panel acepta una rejilla incompleta y el denominador de MASE cambia
uv run python -c "
import pandas as pd, numpy as np
from chronolab.panel import Panel, PanelSpec
from chronolab.evaluation.metrics import seasonal_naive_mae
from chronolab.types import DatasetId
n=96; ds=pd.date_range('2024-01-01', periods=n, freq='h')
y=10+5*np.sin(2*np.pi*np.arange(n)/24)
full=pd.DataFrame({'unique_id':'a','ds':ds,'y':y})
spec=PanelSpec(dataset_id=DatasetId('x'), freq='h', seasonalities=(24,))
print('q completo :', seasonal_naive_mae(full['y'].to_numpy(), season=24))
gapped=full.drop(index=50).reset_index(drop=True)
print('q con hueco:', seasonal_naive_mae(gapped['y'].to_numpy(), season=24))
p=Panel(df=gapped, spec=spec)
print('aceptado. filas=', len(p.df), 'rejilla=', len(p.grid()))"

# C2 — tasa de marcado real de cada detector al alfa publicado
uv run python -c "
import pandas as pd, numpy as np
s=pd.read_parquet('reports/results/anomaly_scores.parquet')
u=s[s['in_support'].fillna(False)]
thr=-np.log10(0.05)
print(u.groupby('detector_id')['score'].apply(lambda x:(x>=thr).mean()).round(4).to_string())"

# C4 / G1 / G10 — el leaderboard publicado
uv run python -c "
import pandas as pd; pd.set_option('display.width',250)
lb=pd.read_parquet('reports/results/leaderboard.parquet')
o=lb[lb['unique_id'].isna()]
print('chronos presente:', 'chronos' in set(o['model_id']))
print('training_regime presente:', 'training_regime' in lb.columns)
print(o[['model_id','n_windows','n_obs','mase','n_obs_prob','coverage_50','coverage_95']].to_string(index=False))"

# G3 — data_gap nunca se evaluó
uv run python -c "
import pandas as pd
t=pd.read_parquet('reports/results/anomaly_results.parquet')
p=t[t['unique_id'].isna() & (t['metric']=='range_recall')]
print(p.pivot_table(index='detector_id', columns='anomaly_type', values='n_events').to_string())"

# G12 — los stubs
wc -l src/chronolab/data/assemble.py src/chronolab/models/wrappers.py \
      src/chronolab/models/registry.py src/chronolab/artifacts/writer.py

# G9 — el job `smoke` que el comentario promete
grep -n "smoke\|^  [a-z-]*:" .github/workflows/ci.yml
```
