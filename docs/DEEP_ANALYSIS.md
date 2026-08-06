# Análisis de los modelos profundos

Generado por `scripts/run_deep_analysis.py` sobre el panel horario sintético de
`tests.fixtures.synthetic` (3 series, ~2000 horas), con el mismo plan de
backtesting que el resto del leaderboard: 6 ventanas de origen rodante
(h=24, `step_size`=24, `holdout_windows`=2) y `refit_every=1`.
Todos los números de este documento son trazables a ese run; ninguno está
escrito a mano.

## 1. Leaderboard completo (holdout, agregado sobre todas las series)

| model_id | mase | mae | rmse | n_params | fit_seconds_mean | predict_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- |
| prophet | 0.2794 | 0.6514 | 0.7978 | — | 1.6351 | 0.3653 |
| nhits | 0.2843 | 0.6627 | 0.8198 | 301,749 | 3.2659 | 0.0666 |
| mstl | 0.2877 | 0.6706 | 0.8513 | — | 3.4400 | 0.0256 |
| lightgbm_direct | 0.3520 | 0.8209 | 1.0353 | — | 46.6535 | 0.0998 |
| lightgbm_recursive | 0.3528 | 0.8225 | 1.0062 | — | 4.0762 | 0.2001 |
| patchtst | 0.3811 | 0.8883 | 1.0935 | 220,136 | 6.6684 | 0.0700 |
| xgboost_recursive | 0.3849 | 0.8973 | 1.1068 | — | 2.4225 | 0.2718 |
| xgboost_direct | 0.3885 | 0.9059 | 1.1430 | — | 23.4004 | 0.1791 |
| seasonal_naive_168 | 0.4255 | 0.9917 | 1.2142 | — | 0.0014 | 0.0015 |
| lstm | 0.4278 | 0.9971 | 1.2091 | 67,304 | 55.7604 | 0.0349 |
| auto_theta | 0.5066 | 1.1795 | 1.5225 | — | 13.9625 | 0.0168 |
| tft | 0.6216 | 1.4498 | 1.7808 | 56,468 | 33.8015 | 0.0958 |
| auto_ets | 0.6533 | 1.5226 | 1.8148 | — | 2.8763 | 0.0118 |
| seasonal_naive | 0.7269 | 1.6933 | 2.0589 | — | 0.0013 | 0.0014 |
| auto_arima | 0.7662 | 1.7843 | 2.1580 | — | 5.1788 | 0.0141 |
| historic_average | 2.3962 | 5.5865 | 6.3562 | — | 0.0011 | 0.0015 |
| drift | 2.4438 | 5.6971 | 6.3408 | — | 0.0012 | 0.0014 |
| naive | 2.4449 | 5.6995 | 6.3433 | — | 0.0012 | 0.0013 |
| window_average | 2.4544 | 5.7218 | 6.4206 | — | 0.0012 | 0.0015 |

Las tres columnas de la derecha son el **eje precisión-coste**, que este hito
añade al leaderboard: `n_params` (nulo en los modelos que no ajustan parámetros
por optimización), segundos medios por ajuste y segundos medios por inferencia.

## 2. Los cuatro modelos del hito

| model_id | mase | mae | rmse | n_params | fit_seconds_mean | predict_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- |
| nhits | 0.2843 | 0.6627 | 0.8198 | 301,749 | 3.2659 | 0.0666 |
| patchtst | 0.3811 | 0.8883 | 1.0935 | 220,136 | 6.6684 | 0.0700 |
| lstm | 0.4278 | 0.9971 | 1.2091 | 67,304 | 55.7604 | 0.0349 |
| tft | 0.6216 | 1.4498 | 1.7808 | 56,468 | 33.8015 | 0.0958 |

- Mejor modelo del leaderboard: **prophet** (MASE 0.2794).
- Mejor modelo profundo: **nhits** (MASE 0.2843).
- El LSTM propio queda en MASE 0.4278 frente al 0.2877 de MSTL: **pierde**, por +0.1401.

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
- **Predicción directa multi-paso**: la cabeza proyecta los 24 pasos de una
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

| feature | value |
| --- | --- |
| observed_target | 0.6266 |
| temp_c | 0.3734 |

**Bloque futuro**

| feature | value |
| --- | --- |
| temp_c | 1.0000 |

**Atención temporal**: el máximo está en el offset 8 respecto al
cutoff, y el 86.4% de la atención total la reciben instantes del
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
