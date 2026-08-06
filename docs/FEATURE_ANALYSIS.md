# Análisis de features de los modelos ML

Generado por `scripts/run_ml_feature_analysis.py` sobre el panel horario
sintético de `tests.fixtures.synthetic` (3 series, ~2000 horas), con un plan
de backtesting de 6 ventanas de origen rodante (h=24,
`step_size`=24, `holdout_windows`=2: 4 de
desarrollo y 2 de reporte). Todos los números de este documento
son trazables a ese run; no hay ninguno escrito a mano.

## 1. Leaderboard (holdout, agregado sobre todas las series)

| model_id | mase | mae | rmse | pinball_mean | fit_seconds_mean |
| --- | --- | --- | --- | --- | --- |
| mstl | 0.2861 | 0.6668 | 0.8391 | nan | nan |
| lightgbm_direct | 0.3520 | 0.8209 | 1.0353 | nan | 47.1424 |
| lightgbm_recursive | 0.3528 | 0.8225 | 1.0062 | nan | 2.5502 |
| xgboost_recursive | 0.3849 | 0.8973 | 1.1068 | nan | 2.7050 |
| xgboost_direct | 0.3885 | 0.9059 | 1.1430 | nan | 26.4760 |
| seasonal_naive_168 | 0.4255 | 0.9917 | 1.2142 | 0.2605 | 0.0042 |
| prophet | 0.4281 | 0.9970 | 1.1961 | 0.2806 | nan |
| seasonal_naive | 0.7269 | 1.6933 | 2.0589 | 0.4575 | 0.0065 |
| historic_average | 2.3962 | 5.5865 | 6.3562 | 1.3748 | 0.0084 |
| auto_theta | 2.4213 | 5.6482 | 5.8581 | nan | nan |
| drift | 2.4438 | 5.6971 | 6.3408 | 1.4831 | 0.0095 |
| naive | 2.4449 | 5.6995 | 6.3433 | 1.4821 | 0.0026 |
| window_average | 2.4544 | 5.7218 | 6.4206 | 1.3782 | 0.0045 |
| auto_ets | 2.8257 | 6.5806 | 8.6399 | nan | nan |
| auto_arima | 4.4527 | 10.3830 | 10.4803 | nan | nan |

La tabla de arriba es la de este hito: quince modelos (seis baselines, cuatro
estadísticos más Prophet, y los cuatro de ML). El
`reports/results/leaderboard.parquet` del repositorio lo regeneró después el
hito de modelos profundos (`docs/DEEP_ANALYSIS.md`) sobre el mismo panel, el
mismo plan y las mismas ventanas de holdout, así que contiene estos quince más
los cuatro profundos —y las cifras de ML de arriba siguen siendo las suyas.

## 2. Estrategia recursiva frente a directa

| model_id | mase | mae | pinball_mean |
| --- | --- | --- | --- |
| lightgbm_direct | 0.3520 | 0.8209 | nan |
| lightgbm_recursive | 0.3528 | 0.8225 | nan |
| xgboost_recursive | 0.3849 | 0.8973 | nan |
| xgboost_direct | 0.3885 | 0.9059 | nan |

- **LightGBM**: la estrategia directa cambia el MASE agregado en
  -0.0008 frente a la recursiva.
- **XGBoost**: la estrategia directa cambia el MASE agregado en
  +0.0036 frente a la recursiva.

### Degradación por paso de horizonte

![Degradación por horizonte](figures/07_horizon_degradation.png)

MASE en el primer paso del horizonte (h_step=1) frente al último (h_step=24):

| model_id | h_step | mase | n_obs |
| --- | --- | --- | --- |
| lightgbm_direct | 1 | 0.3752 | 6 |
| lightgbm_direct | 24 | 0.2166 | 6 |
| lightgbm_recursive | 1 | 0.3752 | 6 |
| lightgbm_recursive | 24 | 0.2250 | 6 |
| xgboost_direct | 1 | 0.3997 | 6 |
| xgboost_direct | 24 | 0.3080 | 6 |
| xgboost_recursive | 1 | 0.3997 | 6 |
| xgboost_recursive | 24 | 0.1775 | 6 |

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

| feature | importance |
| --- | --- |
| lag336 | 1035.0000 |
| lag168 | 1023.0000 |
| lag1__pct_change_lag169 | 858.0000 |
| lag1 | 763.0000 |
| lag1_sub_lag2 | 719.0000 |
| rolling_std_lag1_window_size168 | 643.0000 |
| rolling_std_lag1_window_size24 | 503.0000 |
| lag24 | 454.0000 |
| lag48 | 436.0000 |
| lag2 | 403.0000 |
| lag1_sub_lag25 | 392.0000 |
| temp_c_lag168 | 383.0000 |
| fourier_annual_sin_2 | 381.0000 |
| lag3 | 361.0000 |
| lag1__pct_change_lag25 | 346.0000 |

### SHAP (media de |SHAP| sobre una muestra de 300 filas)

![Importancia SHAP](figures/07_feature_importance_shap.png)

| feature | mean_abs_shap |
| --- | --- |
| lag336 | 4.6038 |
| lag168 | 3.2134 |
| lag1 | 0.6939 |
| lag2 | 0.1133 |
| lag24 | 0.1048 |
| lag48 | 0.0939 |
| fourier_daily_cos_1 | 0.0843 |
| fourier_weekly_sin_1 | 0.0764 |
| lag1__pct_change_lag169 | 0.0731 |
| hour | 0.0402 |
| lag3 | 0.0397 |
| lag1_sub_lag2 | 0.0382 |
| rolling_std_lag1_window_size168 | 0.0323 |
| lag1__pct_change_lag25 | 0.0261 |
| rolling_std_lag1_window_size24 | 0.0257 |

**Acuerdo entre los dos métodos**: 7 de las 10 features más
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

Presupuesto: 8 trials por librería (parámetro `n_trials` de
`chronolab.evaluation.tuning.tune`, configurable), optimizando MASE sobre las
4 ventanas `dev` del plan —nunca sobre las
2 de holdout que se reportan en la sección 1, por construcción
de `chronolab.evaluation.tuning.dev_only_panel` (docs/ARCHITECTURE.md, fuga
L5).

- **LightGBM**: `{'n_estimators': 250, 'learning_rate': 0.048046413497514796, 'num_leaves': 51}`
- **XGBoost**: `{'n_estimators': 150, 'learning_rate': 0.20761410420015303, 'max_depth': 8}`

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
  temperatura está declarada `futr_exog` o `hist_exog`. Con h=24 eso deja
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
