# Hallazgos de la EDA (Fase 3)

> **Fuente:** [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb), ejecutada de principio a fin sin red.
> **Figuras:** [`reports/figures/`](../reports/figures/), 22 archivos con nombres estables `<sección>_<figura>[_<serie>].png`.
> Este documento es lo que leen las fases siguientes; no hace falta reabrir la notebook.

## ⚠️ Esta ejecución usa datos sintéticos, no reales

No hay todavía una descarga real de UCI/REE/Open-Meteo cacheada en `data/raw/` (requiere red, que este entorno no tiene). La notebook usa `chronolab.data.sources.synthetic.SyntheticElectricitySource` y `SyntheticWeatherSource`: tres series de demanda horaria de 14 meses (2023-06-01 a 2024-08-01, Europe/Madrid) con una dificultad de predicción que se pretendía escalonada, más temperatura horaria consistente, generadas con las mismas trampas de DST que tendría una fuente real y con huecos/duplicados/ceros/atípicos inyectados a propósito.

**Los números de este documento describen el pipeline y el método, no demanda eléctrica real.** Sirven para tres cosas: (1) demostrar que cada pieza de `chronolab.data`/`chronolab.viz.plots` funciona end-to-end, (2) fijar la metodología que se reutilizará sin cambios sobre datos reales, y (3) documentar dos problemas metodológicos genuinos que la propia ejecución destapó (§3 y §6) y que aplicarán igual cuando los datos sean reales. Cuando haya red, el primer paso de la Fase 4 debería ser repetir esta notebook con `UCIElectricitySource` y `OpenMeteoSource`/`REEDemandSource` y contrastar si la estructura aquí descrita se sostiene.

---

## 1. Perfil de calidad de datos

`coverage_report` sobre las tres series (10.225 filas crudas cada una, 10.246 tras alinear a rejilla completa):

| serie | cobertura | huecos | duplicados (pares) | ceros | atípicos (z robusto > 4) |
|---|---|---|---|---|---|
| `residential_north` | 99.6 % | 41 | 21 | 0 | 8 |
| `commercial_mixed` | 99.6 % | 41 | 21 | 72 | 80 |
| `volatile_industrial` | 99.6 % | 41 | 21 | 0 | 453 |

Los 72 ceros de `commercial_mixed` son el tramo de 3 días inyectado a propósito (simulando un corte/lectura congelada); las otras dos series no tienen ninguno, como se esperaba de una demanda estrictamente positiva.

**Continuidad tras el cambio de hora — la comprobación central de este apartado.** La ventana cubre el vuelco de otoño real del 29 de octubre de 2023 y el salto de primavera real del 31 de marzo de 2024. `dst_transition_report` compara la trama cruda (en hora local) con la alineada:

| transición | filas locales (3 series) | duplicados en la alineada | huecos en la alineada |
|---|---|---|---|
| 2023-10-29 (vuelco de otoño) | 75 | **0** | **0** |
| 2024-03-31 (salto de primavera) | 70 | **0** | **0** |

Cero duplicados y cero huecos alrededor de ambas transiciones, en las dos direcciones del cambio de hora: `align.to_utc_naive` + `reindex_to_full_grid` resuelven el DST correctamente. (El recuento de filas locales del vuelco de otoño, 75 = 3×25, es exactamente el esperado; el del salto de primavera, 70, no es 3×23=69 porque una imperfección inyectada al azar cayó justo ese día — coincidencia declarada, no error; ver `01_dst_continuity_2024-03-31.png`.)

**Figuras:** `01_quality_overview.png` (barras por serie), `01_series_flags_<serie>.png` (huecos como cortes en la línea, atípicos en rojo), `01_dst_continuity_2023-10-29.png`, `01_dst_continuity_2024-03-31.png`.

---

## 2. Descomposición MSTL (24, 168)

`compute_mstl` sobre las tres series, con reconstrucción aditiva exacta (`trend + seasonal_24 + seasonal_168 + resid == observed`, verificado también por test). Las tres muestran una componente de tendencia y ambas estacionalidades bien separadas del residuo; en `volatile_industrial` el residuo absorbe los eventos de lote inyectados como picos aislados, que es el comportamiento correcto de MSTL — no los reparte entre tendencia y estacionalidad.

**Figuras:** `02_mstl_residential_north.png`, `02_mstl_commercial_mixed.png`, `02_mstl_volatile_industrial.png`.

---

## 3. ACF, PACF y periodograma

### Hallazgo metodológico: la autocorrelación es extremadamente sensible a un puñado de atípicos

Antes de descartar los 8 puntos que `detect_outliers` marcó en `residential_north` (el 0,08 % de 10.246 observaciones), el ACF en el retardo 24 era **0,08**; tras descartarlos, **0,84**. En el retardo 168, de 0,08 a 0,83. Ocho puntos extremos, de diez mil, esconden casi por completo una autocorrelación que en realidad es muy fuerte. Es un efecto más severo que el de §6: la autocorrelación en un retardo concreto depende del producto de dos desviaciones (`(x_t - x̄)(x_{t-k} - x̄)`), así que un solo pico de 8× el valor normal contamina simultáneamente su propio retardo y todos los que lo referencian.

**Consecuencia práctica para la Fase 4:** cualquier diagnóstico basado en ACF/PACF o en periodograma debe correr sobre una serie con los atípicos evidentes tratados (interpolados, capados o excluidos), nunca sobre la serie cruda — de lo contrario, un modelo que en realidad tiene mucha estructura aprovechable parecerá casi un paseo aleatorio.

Con los atípicos descartados (`despike`, reutilizada de §1), sobre `residential_north`:

- ACF(24) = 0,84, ACF(168) = 0,83, PACF(1) = 0,78: autocorrelación fuerte y sostenida, coherente con una serie muy estacional.
- Periodograma: el pico más alto cae en periodo 24h (potencia 14,9), con el segundo armónico en 12h pisándole los talones (14,1) — la forma de doble joroba (mañana + noche) del perfil residencial concentra energía real ahí, no es ruido. El pico semanal en 168h es real pero mucho más débil en potencia absoluta; se ve como un bulto local en la escala logarítmica del gráfico, no como un tercer pico comparable a los de 12h/24h.
- **Con los atípicos presentes, el orden se invertía** (12h por delante de 24h): no solo cambia cuánta estructura se ve, sino cuál periodo parece "el dominante".

**Figuras:** `03_acf_pacf_residential_north.png`, `03_periodogram_residential_north.png`, `03_periodogram_volatile_industrial.png` (espectro visiblemente más plano, sin picos claros en 24/168 — el contraste que se pretendía).

---

## 4. Perfiles agregados

- **Heatmap hora × día de la semana:** `residential_north` muestra el perfil doméstico de doble pico (mañana ~8h, noche ~20-21h) prácticamente igual entre semana y fin de semana; `commercial_mixed` muestra un bloque claro de horario comercial (9-18h entre semana) que casi desaparece el fin de semana (factor 0,30 en el generador) — el contraste de forma entre las dos figuras es nítido.
- **Efecto de festivo:** en el generador, festivo se trata igual que fin de semana; las cajas de `is_holiday=True` se desplazan hacia abajo en ambas series, más marcadamente en `commercial_mixed` (donde el fin de semana ya supone una caída fuerte) que en `residential_north` (donde el efecto de fin de semana es suave, +5 %).
- **Perfil mensual:** en `residential_north` se ve la forma de dos jorobas esperada de la respuesta térmica (invierno y verano por encima del resto del año), superpuesta a la tendencia de crecimiento lento.

**Figuras:** `04_heatmap_hour_dow_residential_north.png`, `04_heatmap_hour_dow_commercial_mixed.png`, `04_holiday_effect_residential_north.png`, `04_holiday_effect_commercial_mixed.png`, `04_monthly_profile_residential_north.png`.

---

## 5. Relación demanda-temperatura

El ajuste LOWESS (no paramétrico, no impone la forma) confirma la **U** esperada en `residential_north`: mínimo en 139,7 (≈21,2 °C, dentro de la zona de confort térmico 18-22 °C del generador), subiendo a ≈147 en el extremo frío (-1,6 °C) y a ≈150 en el extremo cálido (34,5 °C). `commercial_mixed` muestra una U más suave, coherente con una respuesta térmica más débil por diseño (ganancias de calefacción/refrigeración de 0,5/0,6 frente a 0,9/0,7 en residencial).

### Hallazgo metodológico: la correlación cruda de HDD/CDD queda confundida por la tendencia

Grados-día calculados sobre temperatura diaria media (base calefacción 18 °C, base refrigeración 22 °C) y correlacionados con demanda diaria media:

| serie | corr(HDD, demanda) | corr(CDD, demanda) |
|---|---|---|
| `residential_north` | **+0,441** | **−0,231** |
| `commercial_mixed` | +0,223 | −0,199 |

El signo de HDD es el esperado (más frío → más demanda). El de CDD **no**: la respuesta al calor está diseñada para ser positiva, y el ajuste LOWESS la muestra positiva, pero la correlación de Pearson cruda sale negativa en ambas series. La explicación más plausible es que la ventana empieza y termina en verano (2023-06 a 2024-07), y la tendencia de crecimiento a largo plazo queda confundida con la variación estacional de temperatura: no es que la refrigeración reduzca la demanda, es que el CDD alto del primer verano (con la tendencia todavía baja) pesa en la correlación. **La lectura correcta de la respuesta térmica es la curva LOWESS, no la correlación cruda de grados-día**, salvo que esta última se calcule sobre una serie ya sin tendencia ni estacionalidad de baja frecuencia (por ejemplo, sobre el residuo de la descomposición MSTL de §2).

**Figuras:** `05_temp_scatter_residential_north.png`, `05_temp_scatter_commercial_mixed.png`, `05_degree_days_correlation_residential_north.png`, `05_degree_days_correlation_commercial_mixed.png`.

---

## 6. Estadísticos de dificultad de la serie

Igual que en §3, los estadísticos basados en varianza se ven muy distorsionados por los atípicos. Se reporta la versión sin ellos (`despike`) como referencia, con la versión cruda documentada al lado, no oculta:

| serie | fuerza de tendencia | fuerza estacional (24) | fuerza estacional (168) | entropía espectral |
|---|---|---|---|---|
| `residential_north` | 0,674 | 0,868 | 0,248 | 0,427 |
| `commercial_mixed` | 0,771 | 0,895 | **0,777** | **0,324** |
| `volatile_industrial` | 0,096 | 0,204 | 0,158 | 0,889 |

*(con atípicos, sin despicar: 0,040/0,273/0,153/0,932 · 0,278/0,387/0,262/0,776 · 0,066/0,188/0,147/0,927 respectivamente — hasta 10× más bajos donde debería haber más estructura.)*

**`volatile_industrial` es, sin ambigüedad, la más difícil** en las cuatro métricas: fuerza baja en todo, entropía cerca de 1 (espectro casi plano). El diseño pretendía justo eso y los datos lo confirman.

**Lo que el diseño no anticipó:** `commercial_mixed` resulta, según estas cuatro métricas, tan o más estructurada que `residential_north` — mayor fuerza de tendencia (0,771 frente a 0,674), fuerza estacional semanal muy superior (0,777 frente a 0,248) y menor entropía espectral (0,324 frente a 0,427). La intuición de partida ("residencial es el caso fácil de libro de texto") no se sostiene frente a los números. La explicación está en los propios parámetros del generador, no en un error de cálculo: el contraste entre semana y fin de semana de `commercial_mixed` es muy marcado (factor ×0,30 el fin de semana), mientras que el de `residential_north` es suave (×1,05); una modulación semanal casi binaria es, por construcción, mucho más fácil de capturar en la fuerza estacional de periodo 168 que una modulación del 5 %. Es un recordatorio concreto de por qué este documento pide tabular los estadísticos en vez de fiarse de una intuición sobre qué dominio "debería" ser fácil: **la fuerza estacional mide la amplitud relativa del patrón semanal, no si el dominio es residencial o comercial**.

**Interpretación para la Fase 4:** con datos de esta forma, un `SeasonalNaive`/`MSTL` de referencia debería batir con holgura a modelos más complejos en `residential_north` y sobre todo en `commercial_mixed`; en `volatile_industrial`, ningún baseline estacional tiene ninguna ventaja estructural que explotar, y el techo de cualquier modelo va a estar dominado por el ruido y por si el modelo consigue anticipar (que no debería poder, por diseño) los eventos de lote. Es exactamente el papel que en `docs/PLAN_PROYECTO.md` se le pedía a la serie de contraste cripto, cumplido aquí sin depender de una API externa.

**Figuras:** `06_difficulty_table.png` (tabla sin atípicos, la de referencia; la versión con atípicos queda impresa en la notebook, no graficada).

---

## Resumen para quien no lea más que esto

1. **El pipeline de datos funciona de extremo a extremo sin red**, incluyendo el caso más peligroso (DST): cero huecos y cero duplicados alrededor de las dos transiciones reales de la ventana.
2. **Dos hallazgos metodológicos, no anecdóticos, que aplican también a datos reales:** (a) ACF/PACF, periodograma y los estadísticos de dificultad basados en varianza son extremadamente sensibles a un puñado de atípicos — hay que tratarlos antes de calcular cualquiera de los tres, nunca después; (b) la correlación cruda de grados-día con demanda puede salir con el signo equivocado si la ventana de datos confunde tendencia con estacionalidad de temperatura — hay que leer la curva LOWESS, no solo el coeficiente.
3. **La dificultad relativa de una serie no se puede intuir por su dominio**: hubo que medirla, y la medida contradijo la intuición de partida en un caso concreto (`commercial_mixed` resultó más estructurada que `residential_north`).
4. Todo esto es sobre **datos sintéticos**. La acción pendiente, en cuanto haya red, es repetir la misma notebook con las fuentes reales y comprobar cuánto de esto se sostiene.
