# Arquitectura de código — `chronolab`

> **Estado:** propuesta de diseño. Nada implementado todavía.
> **Ámbito:** este documento define la arquitectura de código. Es la referencia normativa
> para todo el trabajo posterior. Complementa a [`PLAN_PROYECTO.md`](PLAN_PROYECTO.md),
> del que discrepa en varios puntos (§10).
> **Autosuficiencia:** quien solo lea este archivo debe poder implementar el proyecto.
> Firmas, invariantes y esquemas están escritos para ser copiados, no interpretados.

---

## 1. Principios de diseño

Seis axiomas. Todo lo demás se deriva de ellos, y cuando haya una duda de diseño no
cubierta aquí, se resuelve invocándolos.

**A1 — El valor del proyecto es el arnés de evaluación, no los modelos.**
Los modelos son commodities. Por tanto el código que se posee (splitter, motor de
backtesting, métricas, artefactos) se escribe y se testea de forma propia; el código de
modelado se envuelve. Corolario: se acepta más complejidad en `evaluation/` que en
`models/`.

**A2 — Ningún `DataFrame` desnudo cruza una frontera de módulo.**
Un `pd.DataFrame` no dice qué es su índice, qué columna es el objetivo, ni cuál de sus
columnas exógenas se conoce a futuro. Todo dato que viaja entre capas va envuelto en un
tipo que declara su contrato (`Panel`, `FutrFrame`, `ScoringFrame`). El *formato* de
Nixtla es un formato de transporte; el *contrato* es el tipo que lo envuelve.

**A3 — Las barreras contra la fuga temporal son estructurales, no disciplinarias.**
"Acuérdate de no hacer X" no es una barrera. Las tres formas de barrera admisibles en
Python, por orden de fuerza:

1. **Ausencia física** — el dato no está en la estructura que recibe el consumidor
   (p. ej. las columnas `hist_exog` no existen en el `FutrFrame`).
2. **Constructor único** — el tipo solo se puede construir por un camino que impone el
   invariante (`FutrFrame` únicamente lo emite un `FutrProvider`).
3. **Aserción en el único camino de código** — no hay una segunda ruta que la evite
   (`predict()` asegura `ds > cutoff` y no existe otra vía para predecir).

Un test es evidencia, no barrera. Un comentario no es nada.

**A4 — Todo resultado publicado es reproducible desde un `run_id`.**
Un `run_id` determina: configuración, semilla, versiones de librerías, SHA de git,
ventanas, vintage de exógenas y artefactos. Si un número aparece en la app o en el
README y no se puede rastrear a un `run_id`, es un bug.

**A5 — La app no calcula nada.**
Streamlit lee artefactos y dibuja. No entrena, no re-puntúa, no agrega métricas que no
estén ya persistidas. Esto no es una preferencia: es lo que permite desplegar sin
`torch` ni `prophet` instalados y arrancar sin red.

**A6 — Los fallos son visibles.**
Un modelo que revienta en una ventana no desaparece del leaderboard: aparece con
`status="failed"`. Un detector que no puede puntuar los primeros `w-1` puntos lo declara
con `scorable=False`. Silenciar es peor que fallar.

---

## 2. Árbol de módulos bajo `src/chronolab/`

Una línea por módulo con su responsabilidad. La regla de dependencias está en §2.1.

```
src/chronolab/
├── __init__.py                  Versión del paquete y re-export de los tipos públicos.
├── py.typed                     Marcador PEP 561: el paquete distribuye tipos.
├── types.py                     NewTypes (RunId, ModelId…), enums (Role, Vintage, Stage). Sin dependencias internas.
├── errors.py                    Jerarquía de excepciones propias (ChronolabError y descendientes).
├── config.py                    Modelos pydantic de conf/*.yaml + resolución de rutas + hashing canónico de configuración.
├── logging.py                   Configuración de logging estructurado y del contexto run_id.
├── panel.py                     Panel y PanelSpec: el contrato interno de datos, sus invariantes y sus proyecciones.
│
├── data/
│   ├── protocols.py             Protocolo DataSource y SourceSpec.
│   ├── align.py                 to_utc_naive(), rejilla regular, completado de huecos, deduplicación, DST.
│   ├── schemas.py               Esquemas pandera de las tramas crudas y del panel canónico.
│   ├── cache.py                 CachedSource: decorador de caché parquet con clave = (source_spec, argumentos).
│   ├── calendar.py              Festivos por país, términos de Fourier, features de calendario local DST-safe.
│   ├── quality.py               Informe de calidad: huecos, duplicados, ceros vs NaN, saltos de DST, quiebres.
│   ├── assemble.py              Ensambla fuentes objetivo + exógenas + estáticas en un Panel validado.
│   ├── futr.py                  Protocolo FutrProvider y sus tres implementaciones (realized / archived / simulated).
│   └── sources/
│       ├── uci_electricity.py   Descarga y parseo de UCI ElectricityLoadDiagrams 2011-2014.
│       ├── ree.py               Cliente de apidatos.ree.es (demanda España, casi tiempo real).
│       ├── open_meteo.py        Cliente de Open-Meteo: archive (realizado) e historical-forecast (vintage).
│       ├── binance.py           Cliente de klines de Binance (serie de contraste).
│       └── synthetic.py         Generador determinista de series para tests y modo demo sin red.
│
├── features/
│   ├── ops.py                   Primitivas exclusivamente retrospectivas: lag, roll, expand, ewm, diff. Ninguna mira adelante.
│   ├── roles.py                 Álgebra de max_lead: propaga la disponibilidad temporal a través de las operaciones.
│   └── builders.py              Conjuntos de features con nombre (calendario, térmicas, lags) usados por los modelos ML.
│
├── models/
│   ├── protocols.py             Protocolos Forecaster y FittedForecaster + ModelRequirements.
│   ├── registry.py              model_id -> fábrica; construye Forecasters a partir de conf/models.yaml y un PanelSpec.
│   ├── baselines.py             Naive, SeasonalNaive, WindowAverage, HistoricAverage en numpy puro (referencia independiente).
│   ├── wrappers.py              ConformalWrapper: convierte cualquier Forecaster puntual en probabilístico.
│   ├── adapters/
│   │   ├── statsforecast.py     Adaptador de statsforecast (AutoARIMA, AutoETS, AutoTheta, MSTL, baselines).
│   │   ├── mlforecast.py        Adaptador de mlforecast (LightGBM, XGBoost), estrategias recursiva y directa.
│   │   ├── neuralforecast.py    Adaptador de neuralforecast (NHITS, TFT, PatchTST).
│   │   ├── prophet.py           Adaptador de Prophet, un ajuste por serie, con festivos y regresores.
│   │   ├── torch_lstm.py        Adaptador del LSTM propio.
│   │   └── chronos.py           Adaptador zero-shot de Chronos-2 / Chronos-Bolt.
│   └── torch/
│       ├── dataset.py           Dataset de ventanas deslizantes y escalado por serie ajustado solo con train.
│       ├── modules.py           Definición del nn.Module del LSTM (encoder + cabeza de cuantiles).
│       └── trainer.py           Bucle de entrenamiento con early stopping, semilla y registro de coste.
│
├── anomaly/
│   ├── protocols.py             Protocolos AnomalyDetector, FittedDetector, Thresholder + ScoringFrame.
│   ├── conformal.py             Detector principal: score conformal sobre residuos fuera de muestra.
│   ├── isolation.py             IsolationForest sobre vector de features de ventana.
│   ├── autoencoder.py           LSTM-Autoencoder: error de reconstrucción como score.
│   ├── matrix_profile.py        Discords con stumpy, sin entrenamiento.
│   ├── thresholds.py            Thresholders: conformal por cuantil, top-k, sigma móvil. Score -> etiqueta.
│   ├── injection.py             Inyección de anomalías sintéticas tipadas + generación del ground truth.
│   └── events.py                Colapsa puntuaciones/etiquetas puntuales en eventos con extensión y severidad.
│
├── evaluation/
│   ├── splitters.py             RollingOriginSplitter: genera Windows. Único emisor de particiones train/test del proyecto.
│   ├── backtest.py              Motor: recorre ventanas × modelos, aplica política de refit, escribe artefactos.
│   ├── metrics.py               MASE, RMSE, MAE, sMAPE, MAPE(guardado), pinball, cobertura empírica, CRPS discreto.
│   ├── anomaly_metrics.py       VUS-PR, F1 por rangos (Tatbul), métricas de afiliación (Huet). Nunca point-adjusted.
│   ├── stats_tests.py           Diebold-Mariano con HAC y corrección HLN; Model Confidence Set para multiplicidad.
│   └── aggregate.py             Reglas de rollup: de forecasts a la tabla metrics. Prohíbe promediar promedios.
│
├── artifacts/
│   ├── schemas.py               Esquemas pandera de cada tabla de artefacto + SCHEMA_VERSION.
│   ├── writer.py                Escritura atómica particionada (.tmp + rename) y manifest.json.
│   └── reader.py                Única ruta de lectura de artefactos. Es la API que consume la app.
│
├── viz/
│   └── plots.py                 Figuras Plotly puras: reciben DataFrames, devuelven Figure. Sin E/S ni estado.
│
├── app/
│   ├── main.py                  Entrada de Streamlit: navegación, selección de run/dataset, estado de sesión.
│   ├── components/              Widgets reutilizables (selector de serie, slider de alfa, tarjeta de métrica).
│   └── pages/
│       ├── 1_overview.py        Serie, descomposición, estadísticos de dificultad, informe de calidad de datos.
│       ├── 2_forecast.py        Predicciones superpuestas con bandas, por serie/modelo/horizonte.
│       ├── 3_leaderboard.py     Tabla de métricas, precisión vs coste, resultados de Diebold-Mariano y MCS.
│       ├── 4_anomalies.py       Series marcadas, slider de alfa, comparativa de detectores, tabla de eventos.
│       └── 5_explainability.py  Importancias, SHAP, atención del TFT, descomposición de la predicción.
│
└── api/
    └── service.py               FastAPI opcional: POST /forecast, POST /anomalies, GET /health. Solo lectura de artefactos.
```

Fuera del paquete:

```
conf/     datasets.yaml · models.yaml · backtest.yaml · anomaly.yaml   (fuente de verdad de la configuración)
scripts/  run_backtest.py · run_anomaly.py · refresh_data.py · build_demo_artifacts.py
tests/    unit/ · property/ · leakage/ · fixtures/
reports/  artifacts/ (subconjunto demo, versionado) · figures/
data/     raw/ · interim/ · processed/ · artifacts/ (runs completos, gitignored)
```

### 2.1 Regla de dependencias entre capas

Las flechas indican "puede importar". Toda arista no dibujada está prohibida.

```
types, errors  ←  (todos)
panel          →  types, errors, data.schemas
data           →  panel, types, config
features       →  panel, types
models         →  panel, features, types
anomaly        →  panel, types, artifacts.reader
evaluation     →  panel, models, anomaly, artifacts.writer, features
artifacts      →  panel, types, config
viz            →  types                      (solo DataFrames y Figure)
app            →  artifacts.reader, viz, config, types
api            →  artifacts.reader, config, types
```

Las dos prohibiciones que importan y por qué:

- **`app` y `api` no pueden importar `models`, `evaluation` ni `anomaly`.** Es la
  materialización de A5. Se hace cumplir con `flake8-tidy-imports` en `ruff`
  (`banned-api`), no con buena voluntad. Efecto secundario deseado: el extra `app` de
  `pyproject.toml` no arrastra `torch`, `prophet` ni `neuralforecast`, lo que hace
  viable el despliegue en Streamlit Community Cloud.
- **`models` no puede importar `evaluation`.** Un modelo no debe poder consultar cómo se
  le va a medir ni qué ventanas existen.

---

## 3. El contrato interno de datos

### 3.1 Los tipos

```python
# src/chronolab/types.py
from enum import StrEnum
from typing import Literal, NewType

RunId     = NewType("RunId", str)      # ULID, ordenable por tiempo
DatasetId = NewType("DatasetId", str)
ModelId   = NewType("ModelId", str)
DetectorId= NewType("DetectorId", str)
SeriesId  = NewType("SeriesId", str)

class Role(StrEnum):
    TARGET      = "target"
    FUTR_EXOG   = "futr_exog"    # conocida a futuro en el instante de predecir
    HIST_EXOG   = "hist_exog"    # solo conocida hasta el cutoff
    STATIC_EXOG = "static_exog"  # constante por serie

class Vintage(StrEnum):
    REALIZED           = "realized"            # valor observado a posteriori: presciencia perfecta
    ARCHIVED_FORECAST  = "archived_forecast"   # la previsión que realmente existía en el cutoff
    SIMULATED_FORECAST = "simulated_forecast"  # realizado + error sintético calibrado por lead

Stage = Literal["dev", "holdout"]
```

```python
# src/chronolab/panel.py
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True, slots=True)
class PanelSpec:
    """Declaración del contenido semántico de un panel.

    Parameters
    ----------
    dataset_id
        Identificador estable del dataset; parte de la clave de los artefactos.
    freq
        Alias de offset de pandas de la rejilla temporal, p. ej. ``"h"``. Es la
        frecuencia *real* de los datos, no un deseo: se valida contra el panel.
    seasonalities
        Longitudes estacionales en pasos, de la más corta a la más larga,
        p. ej. ``(24, 168, 8766)``. ``seasonalities[0]`` es la que usa MASE.
    target
        Nombre de la columna objetivo. Siempre ``"y"`` por convención Nixtla.
    futr_exog, hist_exog, static_exog
        Nombres de columnas por rol. Deben ser disjuntos entre sí y no contener
        ``target``, ``unique_id`` ni ``ds``.
    tz_display
        Zona horaria en la que se presentan los datos al usuario. **No** afecta al
        almacenamiento, que siempre es UTC ingenuo.
    """
    dataset_id: DatasetId
    freq: str
    seasonalities: tuple[int, ...]
    target: str = "y"
    futr_exog: tuple[str, ...] = ()
    hist_exog: tuple[str, ...] = ()
    static_exog: tuple[str, ...] = ()
    tz_display: str = "UTC"

    def __post_init__(self) -> None:
        """Valida disyunción de roles, no vacuidad de seasonalities y freq legal."""

    @property
    def mase_season(self) -> int:
        """Longitud estacional del denominador de MASE."""
        return self.seasonalities[0]

    @property
    def value_columns(self) -> tuple[str, ...]:
        """target + futr_exog + hist_exog, en orden estable."""


@dataclass(frozen=True, slots=True)
class Panel:
    """Panel canónico validado. Único portador de datos entre módulos.

    Invariantes garantizados en construcción (§3.3). Ningún consumidor debe
    volver a comprobarlos; ningún productor puede saltárselos, porque el único
    constructor público es :func:`chronolab.data.assemble.build_panel`.
    """
    df: pd.DataFrame                  # largo: unique_id, ds, *value_columns
    spec: PanelSpec
    static: pd.DataFrame | None = None   # unique_id, *static_exog  (una fila por serie)

    def ids(self) -> tuple[SeriesId, ...]: ...
    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> "Panel":
        """Sub-panel con ``start <= ds <= end``. Conserva spec y static."""
    def train(self, window: "Window") -> "Panel":
        """Rebanada de entrenamiento: ``window.train_start <= ds <= window.cutoff``."""
    def actuals(self, window: "Window") -> pd.DataFrame:
        """unique_id, ds, y para el tramo de evaluación de la ventana."""
    def to_nixtla(self) -> pd.DataFrame:
        """Vista en el dialecto exacto que esperan statsforecast/mlforecast/neuralforecast."""
```

### 3.2 Evaluación crítica de adoptar el formato largo de Nixtla

El plan propone el formato largo `(unique_id, ds, y, *exog)` como formato canónico
interno. **Lo adopto, pero como formato de transporte, no como contrato.** Esa distinción
es la decisión de diseño más importante del documento.

**Lo que se gana — real y grande:**

- Tres familias de modelos (estadística, ML, deep) intercambiables sin código de pegamento.
  Esto no es cosmético: elimina la clase entera de bugs de reindexado y realineación.
- Multi-serie nativo. Un panel de 370 clientes es una tabla, no una lista de 370 objetos.
- Serializa a parquet directamente, particiona bien, y admite validación declarativa con
  `pandera`.
- Se alinea con el vocabulario del ecosistema (`cutoff`, `h`, `X_df`, `futr_df`), lo que
  reduce la fricción de leer la documentación ajena.

**Lo que se pierde — y hay que asumirlo explícitamente:**

1. **El formato no dice nada de lo que importa.** `unique_id, ds, y, temperature,
   holiday, load_lag24` no distingue el objetivo de las exógenas, ni las conocidas a
   futuro de las históricas, ni la frecuencia, ni la zona horaria. Justo las cuatro cosas
   cuya confusión produce fuga de información. **Esto es un defecto del formato, no una
   crítica menor**: el formato es silencioso precisamente donde el proyecto necesita ser
   ruidoso. Mitigación: `PanelSpec` obligatoria, y `Panel` como único portador (A2).

2. **Coste de memoria.** UCI completo a 15 min son ~370 × 140k ≈ 52M filas antes de
   exógenas. En formato largo `ds` se repite una vez por serie y `unique_id` una vez por
   timestamp. Mitigaciones asumidas: `unique_id` como `category`, valores en `float32`,
   `ds` como `datetime64[ns]`, procesado por grupos, parquet con codificación de
   diccionario. Aun así, **este es el motivo real de la decisión D14** (§9): el dataset de
   evaluación es un subconjunto curado de series, no las 370.

3. **La jerarquía se pierde en el `unique_id`.** Región, tipo de cliente o agregado
   nacional acaban codificados dentro de una cadena (`ES.norte.cliente_042`) o en una
   tabla lateral `static`. La primera opción es *stringly-typed* y frágil; la segunda hay
   que mantenerla sincronizada. Consecuencia práctica: la reconciliación jerárquica
   (Fase 12 del plan) necesitará una estructura adicional que el formato largo no da.
   Se documenta como deuda aceptada.

4. **Los huecos son invisibles.** Una hora que falta simplemente no tiene fila, y eso es
   indistinguible de "la serie terminó y empezó otra". Un modelo estacional que asume
   rejilla regular se desalinea en silencio. **Esta es la trampa más peligrosa del formato**
   y es donde muere el proyecto si no se ataja. Mitigación: invariante de rejilla completa
   (§3.3, I3).

5. **La zona horaria y el DST.** El ecosistema Nixtla prefiere `ds` ingenuo. Los datos
   europeos horarios tienen dos horas duplicadas y dos horas ausentes al año. Si se guarda
   hora local ingenua, la estacionalidad de 24 h se rompe dos veces al año y todos los
   resultados quedan contaminados sin síntoma visible. Mitigación: I2 (§3.3).

6. **Las operaciones entre series son incómodas.** Correlaciones cruzadas, modelos
   multivariados y agregaciones requieren pivotar a formato ancho, que solo es seguro si
   todas las series comparten rejilla — otra razón para I3.

**Veredicto:** adoptar el formato largo es correcto y ahorra miles de líneas. Adoptarlo
*a secas*, como propone el plan, es insuficiente: hay que emparejarlo con una
especificación tipada y con tres invariantes duros. Sin eso, se hereda la conveniencia
del ecosistema y también todos sus modos de fallo silencioso.

### 3.3 Invariantes del panel canónico

Se comprueban una sola vez, en `data.assemble.build_panel()`, con `pandera`. A partir de
ahí se asumen. Un `Panel` que exista es un `Panel` válido.

| # | Invariante | Por qué | Qué rompe si no se cumple |
|---|---|---|---|
| **I1** | Columnas exactamente `unique_id`, `ds` y `spec.value_columns`; sin extras, sin faltantes | El esquema es cerrado, no abierto | Columnas fantasma que ningún rol declara acaban usadas como features |
| **I2** | `ds` es `datetime64[ns]` **ingenuo y en UTC**. Una columna tz-aware es error duro | Elimina el DST por construcción: en UTC no hay horas duplicadas ni ausentes | Estacionalidad diaria desalineada dos veces al año |
| **I3** | **Rejilla completa**: para cada `unique_id`, existe una fila por cada punto de la rejilla `freq` entre su primer y último `ds`. Los huecos son filas con `y = NaN`, no filas ausentes | Convierte "falta un dato" en un valor explícito y auditable | Los lags cruzan huecos sin darse cuenta; los modelos estacionales se desfasan |
| **I4** | Ordenado por `(unique_id, ds)`, sin duplicados en esa clave | Requisito de casi todas las librerías; el orden es parte del contrato | Resultados no deterministas |
| **I5** | `y` es `float32`; el resto de columnas numéricas también, salvo booleanas de calendario | Memoria; la precisión extra no significa nada frente al error de predicción | Consumo de RAM ×2 sin beneficio |
| **I6** | Cada serie tiene al menos `max(spec.seasonalities) + 1` observaciones no nulas | Debajo de eso no hay señal estacional que estimar | Modelos que fallan o, peor, que devuelven basura plausible |
| **I7** | Si `static` no es `None`, contiene exactamente una fila por `unique_id` del panel | Evita joins que dupliquen filas silenciosamente | Explosión de filas en el join |

Dos notas sobre I2 que merecen ser explícitas:

- **Ingenuo-UTC en vez de tz-aware-UTC.** Tz-aware sería más seguro en abstracto, pero
  produce fricción constante con el ecosistema Nixtla y con el ida y vuelta a parquet.
  La decisión es ingenuo-por-contrato con una comprobación `pandera` que **rechaza**
  cualquier columna tz-aware. Al ser un error duro, es imposible mezclar los dos
  convenios: o todo el panel es ingenuo-UTC o no hay panel. La conversión ocurre en un
  único punto, `data.align.to_utc_naive()`, y la conversión inversa solo en `viz/` y en
  `app/`, usando `spec.tz_display`.
- **El calendario local sigue siendo local.** "Es festivo", "es lunes" o "es la hora 8"
  son propiedades del tiempo *local*, no del UTC. `data/calendar.py` calcula estas
  features convirtiendo a `tz_display` internamente y devolviendo columnas alineadas al
  índice UTC. Es el único lugar del proyecto donde conviven ambos husos, y está aislado
  a propósito.

---

## 4. Exógenas conocidas a futuro vs históricas

Es el punto donde más proyectos se rompen sin enterarse. Se ataca en cuatro capas
independientes; cada una detendría el error por sí sola.

### 4.1 Capa 1 — Declaración tipada

`PanelSpec.futr_exog` y `PanelSpec.hist_exog` son tuplas disjuntas y congeladas. Se
declaran una vez en `conf/datasets.yaml` y viajan con el panel. Ejemplo:

```yaml
datasets:
  es_demand_h:
    freq: h
    seasonalities: [24, 168, 8766]
    tz_display: Europe/Madrid
    futr_exog: [temp_c, is_holiday, hour_sin, hour_cos, dow, month]
    hist_exog: [voltage, self_consumption]
    static_exog: [client_type, region]
```

La regla mnemotécnica que decide el rol: **¿existiría este valor, con este mismo número,
en un sistema real en el instante `cutoff` para un tiempo posterior a `cutoff`?** El
calendario sí. La previsión meteorológica sí (con matices, §4.3). La temperatura
*observada* del futuro **no**, aunque el fichero histórico la contenga.

### 4.2 Capa 2 — Ausencia física: `FutrFrame`

```python
# src/chronolab/panel.py
@dataclass(frozen=True, slots=True)
class FutrFrame:
    """Exógenas conocidas a futuro para exactamente una ventana.

    Contiene ``unique_id``, ``ds`` y **solo** las columnas de ``spec.futr_exog``.
    Las columnas ``hist_exog`` y ``target`` no están ausentes por convenio: están
    ausentes físicamente. Un modelo no puede leerlas porque no existen en la
    estructura que recibe.

    Solo lo construye un :class:`FutrProvider`. No hay constructor público.
    """
    df: pd.DataFrame
    window: "Window"
    vintage: Vintage
```

Esta es la barrera de tipo *ausencia física* (A3.1) y es la más fuerte del proyecto.
No hay forma de que un adaptador de modelo obtenga el valor futuro de una `hist_exog`
salvo que salga de la API deliberadamente y vuelva a leer el `Panel` — cosa que la
revisión de código detecta trivialmente porque el `Panel` no se le pasa a `predict`.

### 4.3 Capa 3 — Vintage: la trampa que el plan no ve

El plan lista Open-Meteo *archive* como "covariable exógena futura conocida"
(§0, tabla de fuentes). **Esto es incorrecto tal cual está enunciado.** El archivo de
Open-Meteo es reanálisis: la temperatura *revisada a posteriori*. Un sistema real en el
`cutoff` no tenía ese número; tenía una **previsión** meteorológica con error creciente
en el horizonte. Entrenar y evaluar con reanálisis produce un sesgo optimista que
favorece sistemáticamente a los modelos que más peso dan a la exógena — es decir, a
LightGBM y al TFT frente a los baselines. **Es exactamente el tipo de fuga que invalida
la conclusión principal del proyecto y no deja ningún síntoma visible.**

Solución: la exógena futura no es una columna, es una función `(as_of, ds) -> valor`.

```python
# src/chronolab/data/futr.py
class FutrProvider(Protocol):
    """Provee exógenas futuras para una ventana, con semántica de vintage explícita."""

    @property
    def vintage(self) -> Vintage: ...

    def futr(self, window: "Window", ids: Sequence[SeriesId]) -> FutrFrame:
        """Exógenas futuras conocidas en ``window.cutoff`` para el tramo de predicción.

        Parameters
        ----------
        window
            Ventana de backtesting. Define ``cutoff`` (información disponible) y el
            tramo ``[first_pred, last_pred]`` que hay que cubrir.
        ids
            Series para las que se piden exógenas. Deben ser las del train de la ventana.

        Returns
        -------
        FutrFrame
            Exactamente ``len(ids) * window.h`` filas, todas con ``ds > window.cutoff``.
        """
```

Tres implementaciones, en orden de honestidad decreciente:

| Implementación | Vintage | Cuándo se puede usar | Qué significa el resultado |
|---|---|---|---|
| `ArchivedForecastProvider` | `ARCHIVED_FORECAST` | Solo si existe archivo de previsiones para el rango (Open-Meteo *historical-forecast*, ~2021 en adelante) | El número honesto: lo que se sabía en el `cutoff` |
| `SimulatedForecastProvider` | `SIMULATED_FORECAST` | Siempre. Toma el realizado y le suma un error AR(1) cuya desviación crece con el lead, **calibrado sobre el tramo donde sí hay ambas fuentes** | Aproximación defendible cuando no hay archivo |
| `RealizedFutrProvider` | `REALIZED` | Siempre, pero emite `PerfectForesightWarning` | **Cota superior**, no un resultado. Solo válido si se etiqueta como tal |

Consecuencia práctica ineludible: **UCI 2011-2014 no tiene previsiones meteorológicas
archivadas.** Para esa serie hay dos opciones legítimas —`SIMULATED_FORECAST` (recomendada)
o `REALIZED` etiquetado como presciencia perfecta— y ninguna otra. La serie de REE, al ser
reciente, sí admite `ARCHIVED_FORECAST`, y comparar los tres vintages sobre ella es, de
hecho, uno de los hallazgos más presentables del proyecto: *cuánto de la ventaja de los
modelos con exógenas se evapora cuando la exógena es una previsión de verdad.*

**Barrera estructural:** `Vintage` entra en el hash de configuración y se persiste en la
tabla `runs`. `evaluation.aggregate` **rechaza** construir una comparación entre filas con
distinto `futr_vintage`, y la app etiqueta cada leaderboard con su vintage en la cabecera.
No es posible mezclar sin querer.

Nota fina sobre el tramo de entrenamiento: en el train se usa el valor **realizado** de las
`futr_exog`, porque eso es lo que un sistema real tendría del pasado; el vintage solo
aplica al tramo de predicción. Esa asimetría es correcta y hay que implementarla a
propósito: el `Panel` guarda el realizado, y el `FutrProvider` es quien inyecta el vintage
al cruzar el `cutoff`.

### 4.4 Capa 4 — Álgebra de `max_lead`

La distinción binaria futuro/histórico es **insuficiente** para las features derivadas, y
aquí es donde se rompe el 80 % de los proyectos. `lag(y, 24)` no es "histórica" a secas:
es utilizable para predecir a 1..24 pasos y no más allá (sin recursión). La propiedad
correcta es un entero, no un booleano.

> **Definición.** `max_lead(c)` es el mayor adelanto `L ≥ 1` para el que la columna `c`
> es conocida en `cutoff + L` sin recurrir a predicciones propias. `∞` significa "conocida
> en todo el horizonte".

`features/roles.py` propaga `max_lead` a través de cada operación. Nunca se declara a
mano: se calcula.

| Columna o operación | `max_lead` resultante |
|---|---|
| `y` (objetivo) en crudo | `0` — nunca utilizable como feature de un tiempo futuro |
| `hist_exog` en crudo | `0` |
| `futr_exog` en crudo | `∞` |
| Features de calendario derivadas de `ds` | `∞` (son función determinista del tiempo) |
| `lag(c, k)` con `max_lead(c) = 0` | `k` |
| `lag(c, k)` con `max_lead(c) = ∞` | `∞` |
| `lead(c, k)` con `max_lead(c) = ∞` | `∞` |
| `lead(c, k)` con `max_lead(c) = 0` | **`ValueError` en tiempo de construcción** |
| `roll(c, w, shift=k)` (ventana retrospectiva que termina en `t-k`, `k ≥ 1`) | `k` si `max_lead(c)=0`; `∞` si `∞` |
| `roll` centrada o hacia delante | **No existe la operación en `features/ops.py`** |
| `diff(c, k)` | igual que `lag(c, k)` |
| `static_exog` | `∞` |

Reglas de uso que impone `evaluation.backtest`:

- **Estrategia directa** (un modelo por lead `L`): solo se le pasan features con
  `max_lead ≥ L`. El filtrado lo hace el motor, no el adaptador.
- **Estrategia recursiva**: se admiten features con `max_lead < L`, pero **solo** si la
  feature está marcada `recursive_only=True` y el adaptador declara
  `requires.supports_recursive`. `mlforecast` lo implementa correctamente y es el camino
  recomendado; hacerlo a mano es la fuente de bugs número uno.
- Comparar recursiva vs directa sobre el mismo conjunto de features con este filtrado
  automático es un análisis limpio que casi nadie hace bien (Fase 6 del plan).

**Barreras:** (a) `features/ops.py` no exporta ninguna primitiva que mire hacia delante
sobre columnas con `max_lead=0` — la ausencia de la función es la barrera; (b) `lead()`
sobre histórica lanza en construcción, no en ejecución; (c) el test de estabilidad por
prefijos (§8, L3) detecta cualquier operación prospectiva que se cuele por otra vía.

---

## 5. Los tres protocolos centrales

### 5.1 `DataSource`

```python
# src/chronolab/data/protocols.py
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
import pandas as pd
from chronolab.types import Role, SeriesId, Vintage


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Descripción declarativa de lo que produce una fuente.

    Parameters
    ----------
    source_id
        Identificador estable; forma parte de la clave de caché.
    role
        Papel semántico de las columnas que entrega: objetivo, exógena futura,
        exógena histórica o estática. Una fuente tiene un único rol; una fuente
        que produjera columnas de roles distintos se parte en dos.
    value_columns
        Nombres de las columnas de valor que devuelve ``fetch``, sin las claves.
    freq
        Frecuencia nativa de la fuente. Si difiere de la del panel, ``assemble``
        remuestrea explícitamente con una agregación declarada.
    native_tz
        Zona horaria en la que la fuente publica sus marcas de tiempo. Se usa una
        sola vez, en la conversión a UTC ingenuo.
    vintage_aware
        ``True`` si la fuente sabe responder "qué se sabía en ``as_of``". Si es
        ``False``, pasar ``as_of`` es un error, no un parámetro ignorado.
    id_semantics
        Texto libre que documenta qué representa ``unique_id`` en esta fuente
        (cliente, zona de mercado, símbolo…). Va al model card del dataset.
    """
    source_id: str
    role: Role
    value_columns: tuple[str, ...]
    freq: str
    native_tz: str = "UTC"
    vintage_aware: bool = False
    id_semantics: str = ""


class DataSource(Protocol):
    """Contrato de obtención de datos crudos.

    Una fuente **no** limpia, **no** completa huecos y **no** valida invariantes de
    panel: entrega lo que tiene, en formato largo y en UTC ingenuo. La limpieza
    vive en ``data.align`` y el ensamblado en ``data.assemble``. Esta separación es
    deliberada: si cada fuente limpiase a su manera, no habría un solo sitio donde
    auditar el tratamiento de huecos y DST.
    """

    @property
    def spec(self) -> SourceSpec:
        """Descripción declarativa de la fuente. Constante durante su vida."""

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Obtiene datos crudos en formato largo.

        Parameters
        ----------
        start, end
            Intervalo **semiabierto** ``[start, end)`` en UTC ingenuo. La
            semiapertura es obligatoria y uniforme en todo el proyecto: los
            intervalos cerrados son la causa clásica del solape de un punto entre
            train y test.
        ids
            Series a obtener. ``None`` significa "todas las que ofrezca la fuente".
            Los identificadores son los de la fuente; el renombrado canónico ocurre
            en ``assemble``.
        as_of
            Instante de conocimiento: devuelve la información tal y como estaba
            publicada en ese momento. Solo admisible si ``spec.vintage_aware``;
            en caso contrario se lanza :class:`VintageNotSupported`. Nunca se
            ignora en silencio.

        Returns
        -------
        pandas.DataFrame
            Columnas ``unique_id`` (str), ``ds`` (datetime64[ns], UTC ingenuo) y
            ``spec.value_columns``. Puede tener huecos, puede no estar ordenado y
            puede traer duplicados: son problema de ``align``, no del llamante.

        Raises
        ------
        VintageNotSupported
            Si se pasa ``as_of`` a una fuente con ``vintage_aware=False``.
        SourceUnavailable
            Si la fuente remota no responde. El decorador de caché la captura y
            sirve la última versión válida marcándola como obsoleta.
        """
```

Notas de diseño:

- **La caché no está dentro de las fuentes**, sino en `CachedSource`, un decorador que
  implementa `DataSource` y envuelve a otro. Así cada fuente es trivialmente testeable y
  la política de invalidación se cambia en un solo sitio.
- `fetch` debe ser **idempotente respecto a sus argumentos** para que la caché sea
  correcta. Las fuentes con datos revisables (REE revisa demanda) deben ser
  `vintage_aware` o declarar su ventana de revisión en la clave de caché.
- Una fuente que produce a la vez objetivo y exógenas se parte en dos objetos. Forzar
  `role` a ser único evita la abstracción que se filtra: obligar a Open-Meteo a devolver
  una columna `y` que luego se renombra es exactamente el tipo de conveniencia que
  después nadie sabe interpretar.

### 5.2 `Forecaster`

Este es el protocolo que tiene que envolver por igual `statsforecast`, `mlforecast`,
`neuralforecast`, Prophet, un LSTM propio y Chronos zero-shot. Tres decisiones lo hacen
posible sin que la abstracción se filtre:

1. **`fit` devuelve un objeto nuevo**, no muta `self`. Un `Forecaster` es una
   *configuración*; un `FittedForecaster` es *configuración + información hasta un
   cutoff*. Chronos, que no entrena, implementa `fit` capturando el contexto: es un caso
   degenerado legítimo, no un parche.
2. **`h` es un parámetro de `fit`, no de `predict`.** `neuralforecast` y PatchTST
   necesitan el horizonte para construir la red; la estrategia directa necesita saber
   cuántos modelos entrenar. Y elimina de raíz la posibilidad de que `h` de entrenamiento
   y `h` de predicción no coincidan.
3. **El objeto ajustado conoce su propio `cutoff`** y `predict` lo verifica. Es la
   aserción de A3.3 y aplica igual a los seis backends.

```python
# src/chronolab/models/protocols.py
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
import pandas as pd
from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId

# Rejilla canónica de cuantiles del proyecto. Fijada en conf/backtest.yaml.
QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)


@dataclass(frozen=True, slots=True)
class ModelRequirements:
    """Capacidades y necesidades declaradas de un modelo.

    El motor de backtesting las lee para decidir qué pasarle, qué pedirle, cuándo
    reajustarlo y qué ventanas saltarse. Un modelo que declara mal sus requisitos
    falla rápido y ruidosamente en la primera ventana, no en silencio.
    """
    needs_futr_exog: bool = False      # si True y no hay FutrProvider, el run aborta
    uses_hist_exog: bool = False
    uses_static_exog: bool = False
    supports_quantiles: bool = False   # si False, las columnas q_* se escriben como NaN
    supports_recursive: bool = False   # admite features con max_lead < L vía recursión
    min_context: int = 1               # pasos mínimos de train; ventanas cortas se saltan
    handles_nan_target: bool = False   # si False, el adaptador imputa dentro de fit
    is_zero_shot: bool = False         # aparece marcado como tal en el leaderboard
    refit_cost: Literal["free", "cheap", "expensive"] = "cheap"


class Forecaster(Protocol):
    """Configuración de un modelo de predicción. Inmutable y sin estado ajustado."""

    @property
    def model_id(self) -> ModelId:
        """Identificador estable. Clave de partición de los artefactos."""

    @property
    def requires(self) -> ModelRequirements:
        """Capacidades declaradas. Constante."""

    def fit(self, train: Panel, *, h: int) -> "FittedForecaster":
        """Ajusta el modelo con **exclusivamente** los datos de ``train``.

        Parameters
        ----------
        train
            Rebanada de entrenamiento, ya recortada por el motor a
            ``ds <= window.cutoff``. Es un :class:`Panel` completo: lleva su
            ``spec``, sus exógenas históricas y futuras (valores realizados del
            pasado) y sus estáticas. El modelo **no** recibe la ventana ni el
            panel completo, así que no tiene forma de mirar más allá del cutoff.
        h
            Horizonte en pasos de ``spec.freq``. Fijo para toda la vida del objeto
            ajustado.

        Returns
        -------
        FittedForecaster
            Objeto nuevo. ``self`` no se modifica: el mismo ``Forecaster`` puede
            ajustarse en muchas ventanas en paralelo sin contaminación cruzada.

        Notes
        -----
        Todo el preprocesado dependiente de datos —escalado, imputación, selección
        de features, calibración conformal interna— ocurre **aquí dentro**, con lo
        que por construcción se ajusta solo con train. El proyecto no tiene una
        etapa global de preprocesado, y ese hueco en la arquitectura es
        intencionado (§8, L2).
        """


class FittedForecaster(Protocol):
    """Modelo ajustado hasta un instante concreto. Inmutable."""

    @property
    def model_id(self) -> ModelId: ...

    @property
    def cutoff(self) -> pd.Timestamp:
        """Última marca de tiempo **incluida** en el entrenamiento.

        Es la frontera de información del objeto. ``predict`` la usa para verificar
        que nada de lo que se predice cae en el pasado conocido.
        """

    @property
    def h(self) -> int:
        """Horizonte con el que se ajustó."""

    @property
    def fit_seconds(self) -> float:
        """Coste de ajuste medido. Se persiste para el eje precisión/coste."""

    @property
    def n_params(self) -> int | None:
        """Número de parámetros entrenables, o ``None`` si no aplica."""

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        """Predice ``h`` pasos hacia delante para todas las series del train.

        Parameters
        ----------
        futr
            Exógenas conocidas a futuro del tramo a predecir, emitidas por un
            :class:`FutrProvider`. Obligatorio si ``requires.needs_futr_exog``.
            Contiene solo columnas ``futr_exog``: el modelo no puede acceder a
            exógenas históricas del futuro porque no están en la estructura.
        quantiles
            Cuantiles a estimar, en ``(0, 1)``. Los modelos con
            ``supports_quantiles=False`` devuelven ``NaN`` en esas columnas; nunca
            un intervalo inventado.

        Returns
        -------
        pandas.DataFrame
            Exactamente ``n_series * h`` filas. Columnas: ``unique_id``, ``ds``,
            ``y_hat`` y una por cuantil (``q_0250``, ``q_5000``…, §7.3).
            Todas las ``ds`` cumplen ``ds > cutoff`` y caen en la rejilla de
            ``spec.freq``.

        Raises
        ------
        CutoffViolation
            Si alguna ``ds`` de ``futr`` o de la salida es ``<= cutoff``. Es la
            aserción central anti-fuga del proyecto y se comprueba siempre, también
            en producción: su coste es despreciable frente a su valor.
        MissingFutrExog
            Si ``requires.needs_futr_exog`` y ``futr is None``.
        """
```

Cómo encaja cada backend en esta firma —la prueba de que no se filtra:

| Backend | `fit` | `predict` | Observaciones |
|---|---|---|---|
| `statsforecast` | Construye `StatsForecast` y llama `.fit(train.to_nixtla())` | `.predict(h, X_df=futr.df)` | `refit_cost="cheap"`; cuantiles nativos vía `level`, convertidos a rejilla canónica |
| `mlforecast` | `MLForecast(...).fit(...)` con `lags`/`lag_transforms` filtrados por `max_lead` | `.predict(h, X_df=futr.df)` | Único backend con `supports_recursive=True` |
| `neuralforecast` | Construye la red **dentro de `fit`**, con `h`, `futr_exog_list`, `hist_exog_list` derivados de `train.spec` | `.predict(futr_df=futr.df)` | `refit_cost="expensive"`; por eso la construcción diferida es necesaria, no un capricho |
| Prophet | Un `Prophet()` por serie, `add_regressor` por cada `futr_exog` | `predict` sobre el frame futuro por serie | Coste O(n_series) por ventana: ver riesgo R2 |
| LSTM propio | Dataset de ventanas + escalado por serie ajustado con train, early stopping sobre un corte interno del propio train | Forward pass con cabeza de cuantiles | `n_params` real, se reporta |
| Chronos | **No entrena**: guarda el contexto (últimos `max(min_context, ctx_len)` puntos de train) y el `cutoff` | `pipeline.predict(context, h)` → cuantiles nativos | `is_zero_shot=True`, `refit_cost="free"`; `fit_seconds ≈ 0` y se reporta como tal |

El caso Chronos es el que valida el diseño: un modelo sin entrenamiento encaja sin
condicionales especiales en el motor, porque "ajustar" se ha definido como "fijar la
frontera de información", que es la operación que **todos** comparten.

### 5.3 `AnomalyDetector`

```python
# src/chronolab/anomaly/protocols.py
from dataclasses import dataclass
from typing import Literal, Protocol
import pandas as pd
from chronolab.panel import PanelSpec
from chronolab.types import DetectorId, ModelId


@dataclass(frozen=True, slots=True)
class ScoringFrame:
    """Tramo de serie con predicciones **fuera de muestra** alineadas.

    Es la única entrada de los detectores. Lo construye exclusivamente
    ``artifacts.reader.scoring_frame()`` a partir de la tabla ``forecasts`` de un
    run, es decir, a partir de predicciones que por construcción son fuera de
    muestra. Un detector no puede recibir predicciones in-sample porque no existe
    ningún camino de código que se las entregue.

    Attributes
    ----------
    df
        ``unique_id``, ``ds``, ``y``, ``y_hat``, columnas ``q_*`` y, opcionalmente,
        las exógenas del panel. Rejilla completa (I3), ordenada (I4).
    model_id
        Modelo del que provienen ``y_hat`` y los cuantiles. ``None`` para
        detectores que no usan predicción; forma parte del ``detector_id`` efectivo
        en los artefactos, porque "IsolationForest sobre residuos de MSTL" y
        "…sobre residuos de NHITS" son detectores distintos.
    start, end
        Extremos inclusivos del tramo cubierto.
    """
    df: pd.DataFrame
    spec: PanelSpec
    model_id: ModelId | None
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True, slots=True)
class DetectorRequirements:
    needs_forecast: bool = False    # requiere y_hat no nulo
    needs_quantiles: bool = False   # requiere columnas q_* no nulas
    window: int = 1                 # puntos de contexto por score; determina el warm-up
    needs_calibration: bool = True  # False para métodos sin ajuste (MatrixProfile)
    fit_cost: Literal["free", "cheap", "expensive"] = "cheap"


class AnomalyDetector(Protocol):
    """Configuración de un detector. Inmutable y sin estado calibrado."""

    @property
    def detector_id(self) -> DetectorId: ...

    @property
    def requires(self) -> DetectorRequirements: ...

    def fit(self, calib: ScoringFrame) -> "FittedDetector":
        """Calibra el detector con un tramo **anterior** al que se puntuará.

        Parameters
        ----------
        calib
            Tramo de calibración. Para el detector conformal son los residuos de
            calibración que definen el cuantil; para IsolationForest y el
            autoencoder es el conjunto de ajuste; los métodos sin calibración
            (``needs_calibration=False``) solo lo usan para fijar su ``cutoff``.

        Returns
        -------
        FittedDetector
            Con ``cutoff = calib.end``.
        """


class FittedDetector(Protocol):
    """Detector calibrado hasta un instante concreto."""

    @property
    def detector_id(self) -> DetectorId: ...

    @property
    def cutoff(self) -> pd.Timestamp:
        """Último instante usado en calibración. ``score`` exige ``ds > cutoff``."""

    def score(self, frame: ScoringFrame) -> pd.DataFrame:
        """Puntúa cada marca de tiempo del tramo.

        Parameters
        ----------
        frame
            Tramo a puntuar. Debe cumplir ``frame.start > cutoff``.

        Returns
        -------
        pandas.DataFrame
            Una fila por ``(unique_id, ds)`` de la entrada, sin excepción.
            Columnas:

            ``score`` : float32
                Grado de anomalía. **Mayor = más anómalo.** Es una magnitud
                **ordinal dentro de un par (detector, serie)** y no es comparable
                entre detectores ni entre series. No se exige calibración a una
                escala común, porque exigirla sería falso: el error de
                reconstrucción de un autoencoder y un p-valor conformal no viven en
                la misma escala, y forzarlos a ella es precisamente lo que hace que
                las comparativas de detectores publicadas no signifiquen nada.
            ``scorable`` : bool
                ``False`` en el calentamiento (primeros ``window - 1`` puntos) o
                donde ``y`` es ``NaN``. Donde es ``False``, ``score`` es ``NaN``.

        Raises
        ------
        CutoffViolation
            Si ``frame.start <= cutoff``.
        """
```

Dos consecuencias de diseño que se derivan de esta firma:

- **Puntuar y etiquetar son operaciones distintas.** `score` no devuelve booleanos.
  El umbralizado vive en `anomaly/thresholds.py` bajo un protocolo aparte
  (`Thresholder.fit(calib_scores) -> FittedThresholder.threshold(alpha)`). Motivo: VUS-PR
  y las curvas PR necesitan el score continuo, mientras que F1 por rangos necesita
  etiquetas; si el detector devolviera etiquetas se perdería irreversiblemente la
  información que exige la métrica principal. Además, así el slider de α de la app es una
  búsqueda en tabla y no un recálculo.
- **El evaluador iguala la máscara `scorable` entre detectores antes de comparar.**
  Un detector con ventana de 512 puntos puntúa menos instantes que uno de ventana 1; si
  cada uno se evalúa sobre su propio soporte, el de ventana larga sale artificialmente
  favorecido porque se salta el arranque de la serie. `evaluation.anomaly_metrics` calcula
  la intersección de máscaras de todos los detectores comparados y evalúa a todos sobre
  ella. Esto es un detalle pequeño con efecto grande y casi nadie lo hace.

---

## 6. Motor de backtesting

### 6.1 `Window` y el splitter

```python
# src/chronolab/evaluation/splitters.py
@dataclass(frozen=True, slots=True)
class Window:
    """Una ventana de origen rodante. Inmutable y autoconsistente.

    Attributes
    ----------
    window_id
        Índice 0-based, creciente en el tiempo.
    stage
        ``"dev"`` para ventanas de desarrollo (tuning, selección) o ``"holdout"``
        para las de reporte. Ver §8, L5.
    train_start, cutoff
        Extremos **inclusivos** del tramo de entrenamiento. ``cutoff`` es la
        frontera de información de la ventana.
    first_pred, last_pred
        Extremos **inclusivos** del tramo de evaluación.
        ``first_pred = cutoff + (gap + 1) * freq`` y
        ``last_pred = first_pred + (h - 1) * freq``.
    h, gap
        Horizonte y separación en pasos.
    """
    window_id: int
    stage: Stage
    train_start: pd.Timestamp
    cutoff: pd.Timestamp
    first_pred: pd.Timestamp
    last_pred: pd.Timestamp
    h: int
    gap: int


class RollingOriginSplitter:
    """Genera ventanas de origen rodante. **Único emisor de particiones del proyecto.**

    Parameters
    ----------
    h
        Horizonte de predicción en pasos.
    n_windows
        Número de ventanas.
    step_size
        Separación entre cutoffs consecutivos, en pasos. Si ``step_size < h`` las
        ventanas de evaluación se solapan; se permite, pero el solape se registra
        porque afecta a la independencia que asume el test de Diebold-Mariano.
    gap
        Pasos descartados entre ``cutoff`` y ``first_pred``. Sirve para emular
        latencia de datos y para cortar la autocorrelación de corto alcance.
    mode
        ``"expanding"`` (train crece) o ``"sliding"`` (train de longitud fija).
    train_size
        Longitud del train en pasos. Obligatorio en ``"sliding"``.
    holdout_windows
        Número de ventanas finales marcadas ``stage="holdout"``. Las anteriores son
        ``"dev"``.
    min_context
        Ventanas cuyo train sea más corto se descartan con aviso, no se recortan.
    """

    def split(self, panel: Panel) -> tuple[Window, ...]:
        """Genera las ventanas por aritmética sobre la rejilla regular del panel.

        No existe ninguna sobrecarga que acepte máscaras booleanas, índices
        arbitrarios ni fechas sueltas. Es deliberado: si la única forma de obtener
        una partición es esta función, no hay forma de escribir un split aleatorio
        por accidente (§8, L1).
        """
```

Nota sobre el solape: con `step_size < h` las ventanas comparten instantes evaluados y
los diferenciales de pérdida están correlacionados. `stats_tests` lo tiene en cuenta
usando estimación HAC con retardo `gap + h - 1`; y la tabla `windows` persiste
`step_size` para que la app pueda advertirlo.

### 6.2 El bucle del motor

`evaluation/backtest.py` es el corazón del proyecto. Su estructura, en pseudocódigo
fiel al orden real de operaciones:

```
plan     = BacktestPlan.from_config(conf)          # h, gap, n_windows, quantiles, refit
panel    = build_panel(dataset_cfg)                # invariantes I1..I7 verificados aquí
windows  = RollingOriginSplitter(**plan).split(panel)
futr     = make_futr_provider(dataset_cfg.vintage) # realized | archived | simulated
run_id   = new_run(panel.spec, plan, futr.vintage) # escribe runs.parquet + windows.parquet

for model_id in plan.models:
    forecaster = registry.build(model_id, spec=panel.spec, seed=plan.seed, params=...)
    fitted     = None
    for w in windows:
        if len(train) < forecaster.requires.min_context:  skip(w, "short_train"); continue
        if needs_refit(w, forecaster.requires.refit_cost, plan.refit_every):
            fitted = forecaster.fit(panel.train(w), h=w.h)     # ← solo ve ds <= w.cutoff
        futr_frame = futr.futr(w, ids=panel.ids())             # ← solo columnas futr_exog
        try:
            pred = fitted.predict(futr_frame, quantiles=plan.quantiles)  # ← asegura ds > cutoff
        except Exception as exc:
            record_model_run(status="failed", error=exc); continue
        pred = repair_quantile_crossing(pred)                  # ordena y registra si hubo cruce
        write_forecasts(run_id, model_id, w, join_actuals(pred, panel))
        record_model_run(run_id, model_id, w, fit_seconds, predict_seconds, n_params, "ok")

aggregate_metrics(run_id)      # forecasts -> metrics.parquet  (nunca promedia promedios)
run_dm_tests(run_id)           # solo sobre stage == "holdout"
finalize(run_id)               # escribe manifest.json el último: hasta entonces el run es invisible
```

Cinco propiedades que se siguen de este bucle y que conviene tener presentes:

- **El modelo nunca ve el panel completo.** Solo `panel.train(w)`, que es un `Panel` ya
  recortado. La API no ofrece otra cosa.
- **Política de refit explícita y registrada.** `refit_every` se guarda por
  `(run, model)`. Reutilizar un ajuste en ventanas posteriores no es fuga —es
  obsolescencia— pero cambia el resultado, así que se reporta. Los modelos
  `refit_cost="expensive"` usan por defecto `refit_every = n_windows` (un solo ajuste) y
  eso aparece en el leaderboard.
- **Los fallos ocupan una fila.** Un modelo que revienta en 3 de 20 ventanas se ve; sus
  métricas se calculan solo sobre las ventanas con `status="ok"` y `n_obs` lo delata.
- **El cruce de cuantiles se repara y se cuenta.** Ordenar los cuantiles es correcto
  (proyección isotónica trivial), pero la frecuencia del cruce es un diagnóstico del
  modelo y se persiste como métrica.
- **El run es atómico.** Sin `manifest.json`, `artifacts.reader` ignora el directorio.
  Streamlit no puede leer un run a medio escribir, ni siquiera si el cron de GitHub
  Actions está escribiendo en ese instante.

### 6.3 Por qué un motor propio y no `cross_validation` de Nixtla

Es mi discrepancia principal con el plan (§1: *"Backtesting (`cross_validation`) unificado
y ya probado"*). Cuatro razones, en orden de peso:

1. **La tesis del proyecto es que el arnés de evaluación es lo valioso (A1).** Delegar el
   arnés en una librería y luego afirmar que el mérito del repo es el rigor del arnés es
   contradictorio. Lo que se posee es lo que se puede defender.
2. **No cubre los seis backends.** Prophet con festivos propios, el LSTM en PyTorch y
   Chronos zero-shot quedan fuera; habría que escribir un segundo camino de backtesting
   para ellos y entonces existen dos protocolos de evaluación distintos —el peor de los
   mundos, porque los números dejan de ser comparables.
3. **`gap` y la semántica de `refit` no son homogéneas** entre `statsforecast`,
   `mlforecast` y `neuralforecast`. Un `gap` uniforme es requisito de los tests de fuga.
4. **Las garantías anti-fuga serían suyas, no verificables por mí.** El criterio de
   calidad "hay al menos un test que detectaría la fuga si la introdujeses" (§6 del plan)
   exige poder introducirla, y para eso hay que ser dueño del splitter.

Coste asumido y declarado: se pierde la paralelización interna en Numba que
`statsforecast.cross_validation` hace sobre las ventanas. Se compensa paralelizando en el
eje `(modelo, ventana)` con `joblib`, que es un eje más grueso pero suficiente. Estimación
del motor propio: ~150 líneas el splitter, ~250 el bucle. Es un coste pequeño para lo que
compra.

---

## 7. Esquema de artefactos

Objetivo: que la app Streamlit dibuje cualquier panel sin entrenar nada, sin unir tablas
grandes en caliente y sin recalcular métricas. Es un esquema en estrella pequeño, en
parquet, particionado en Hive.

### 7.1 Alcance de un run

**Un `run` = un dataset × un plan de backtesting × un vintage de exógenas.** Comparar
modelos entre sí solo es legítimo dentro de un run. Esta restricción simplifica todo el
esquema y elimina la clase de errores "comparé un MASE calculado con otras ventanas".

### 7.2 Disposición en disco

```
<ARTIFACT_ROOT>/
├── runs.parquet                                     índice global de runs (append-only)
├── datasets/dataset_id=<d>/
│   ├── panel.parquet                                panel canónico congelado del dataset
│   ├── panel_meta.json                              PanelSpec serializada + informe de calidad
│   └── anomaly_truth.parquet                        ground truth (independiente del run)
└── runs/run_id=<r>/
    ├── manifest.json                                ESCRITO EL ÚLTIMO. Sin él, el run no existe
    ├── windows.parquet
    ├── model_runs.parquet
    ├── metrics.parquet
    ├── dm_tests.parquet
    ├── anomaly_thresholds.parquet
    ├── anomaly_events.parquet
    ├── explanations.parquet
    ├── forecasts/model_id=<m>/part-0000.parquet
    └── anomaly_scores/detector_id=<k>/part-0000.parquet
```

Elecciones de almacenamiento y su motivo:

- **Partición por `model_id` / `detector_id`.** El argumento decisivo no es la lectura
  sino la **reescritura idempotente**: volver a lanzar un solo modelo reescribe un solo
  directorio sin tocar los demás. Secundariamente, la app carga modelos bajo demanda.
- **`unique_id` no es clave de partición.** Con cientos de series produciría un enjambre
  de ficheros diminutos. En su lugar, las filas se ordenan por `(unique_id, window_id, ds)`
  dentro de cada fichero y se fijan grupos de fila de ~200 k filas, de modo que el
  *predicate pushdown* de pyarrow filtre por serie leyendo solo los grupos necesarios.
- **Compresión `zstd` nivel 3**, diccionario activado para las columnas de identificador.
- **Escritura atómica**: todo el run se escribe en `runs/.tmp-<run_id>/` y se renombra al
  final; `manifest.json` es lo último. `artifacts.reader` ignora directorios sin manifest.
  Esto es lo que hace seguro el cron de `refresh-data.yml` leyendo y escribiendo a la vez.
- **`schema_version`** vive en el manifest y en `runs.parquet`. El lector **rechaza**
  versiones desconocidas en lugar de interpretar mal columnas ausentes.

### 7.3 Convención de columnas de cuantil

Nombre `q_<int>` donde `<int> = round(quantile * 10000)` con relleno a 4 dígitos:

| cuantil | 0.025 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.975 |
|---|---|---|---|---|---|---|---|
| columna | `q_0250` | `q_1000` | `q_2500` | `q_5000` | `q_7500` | `q_9000` | `q_9750` |

Motivos: sin puntos ni signos (compatible con cualquier motor SQL sobre parquet), orden
lexicográfico = orden numérico, y sin ambigüedad `lo`/`hi`.

**Los cuantiles son la representación canónica, no los intervalos `lo`/`hi`.** La pérdida
pinball y el CRPS discreto se definen sobre cuantiles; un intervalo al 95 % es un par de
cuantiles, pero no todo conjunto de cuantiles es una familia de intervalos simétricos.
Guardar `lo`/`hi` obliga a asumir simetría, que es falsa en demanda eléctrica. La
conversión a bandas para dibujar se hace en `viz/`.

### 7.4 Tablas

#### `runs.parquet` — una fila por run

| Columna | Tipo | Descripción |
|---|---|---|
| `run_id` | `string` | ULID, ordenable por tiempo de creación |
| `created_at` | `timestamp[us]` | UTC |
| `dataset_id` | `string` | Dataset del run |
| `schema_version` | `string` | SemVer del esquema de artefactos |
| `git_sha` | `string` | Commit del código que lo generó |
| `git_dirty` | `bool` | `True` si el árbol tenía cambios sin commitear (resultado no reproducible) |
| `config_hash` | `string` | SHA-256 del JSON canónico de la configuración efectiva |
| `seed` | `int32` | Semilla global |
| `futr_vintage` | `string` | `realized` / `archived_forecast` / `simulated_forecast` |
| `h`, `gap`, `n_windows`, `step_size` | `int16` | Plan de backtesting |
| `mode` | `string` | `expanding` / `sliding` |
| `freq` | `string` | Frecuencia del panel |
| `quantiles` | `string` | JSON de la rejilla de cuantiles usada |
| `lib_versions` | `string` | JSON `{paquete: versión}` de las librerías de modelado |
| `hardware` | `string` | CPU/GPU y RAM, para contextualizar los tiempos |
| `notes` | `string` | Texto libre |

#### `windows.parquet` — una fila por ventana

| Columna | Tipo | Descripción |
|---|---|---|
| `run_id` | `string` | |
| `window_id` | `int16` | 0-based, creciente en el tiempo |
| `stage` | `string` | `dev` / `holdout` |
| `train_start`, `cutoff` | `timestamp[ns]` | Extremos inclusivos del train |
| `first_pred`, `last_pred` | `timestamp[ns]` | Extremos inclusivos de la evaluación |
| `n_train_obs` | `int64` | Observaciones no nulas de `y` en el train |
| `n_series` | `int32` | Series con contexto suficiente en esta ventana |

#### `model_runs.parquet` — una fila por (modelo, ventana)

| Columna | Tipo | Descripción |
|---|---|---|
| `run_id`, `model_id` | `string` | |
| `window_id` | `int16` | |
| `status` | `string` | `ok` / `failed` / `skipped` |
| `error` | `string` | Tipo y mensaje si `failed`; `null` si no |
| `refit` | `bool` | `True` si se reajustó en esta ventana |
| `refit_every` | `int16` | Política aplicada |
| `fit_seconds`, `predict_seconds` | `float32` | Coste medido; `0` para reutilizaciones |
| `n_params` | `int64` | Parámetros entrenables; `null` si no aplica |
| `peak_rss_mb` | `float32` | Pico de memoria residente |
| `is_zero_shot` | `bool` | Copiado de `ModelRequirements` para no tener que unir |
| `quantile_crossings` | `int32` | Nº de reparaciones de cruce de cuantiles |

#### `forecasts/model_id=<m>/*.parquet` — la tabla de hechos

| Columna | Tipo | Descripción |
|---|---|---|
| `unique_id` | `string` (dict) | Serie |
| `window_id` | `int16` | Clave de unión con `windows` |
| `cutoff` | `timestamp[ns]` | Desnormalizado: evita la unión al dibujar |
| `ds` | `timestamp[ns]` | Instante predicho, UTC ingenuo |
| `h_step` | `int16` | `1..h`, relativo a `first_pred`. El adelanto real es `gap + h_step` |
| `y` | `float32` | Valor observado. **Desnormalizado a propósito** (§7.5). `NaN` si hueco |
| `y_hat` | `float32` | Predicción puntual |
| `q_0250` … `q_9750` | `float32` | Cuantiles. `NaN` si el modelo no los soporta |

Clave lógica: `(run_id, model_id, unique_id, window_id, ds)`. Orden físico:
`(unique_id, window_id, ds)`.

**No se almacena el residuo.** Es `y - y_hat`, cuesta nada y almacenarlo crea una segunda
fuente de verdad que puede desincronizarse. Lo mismo con `lead`, derivable de `h_step` y
`gap`.

#### `metrics.parquet` — largo, con marginalización explícita

| Columna | Tipo | Descripción |
|---|---|---|
| `run_id`, `model_id` | `string` | |
| `unique_id` | `string` \| `null` | `null` = agregado sobre series |
| `window_id` | `int16` \| `null` | `null` = agregado sobre ventanas |
| `h_step` | `int16` \| `null` | `null` = agregado sobre el horizonte |
| `stage` | `string` | `dev`, `holdout` o `all` |
| `metric` | `string` | `mase`, `rmse`, `mae`, `smape`, `mape`, `pinball_q0250`, `coverage_95`, `crps_discrete`, `mase_denominator`, … |
| `value` | `float64` | |
| `n_obs` | `int64` | Observaciones que entraron en el cálculo |

Formato largo porque las métricas son heterogéneas y se añaden a lo largo del proyecto:
una tabla ancha obligaría a migrar el esquema cada vez. `null` en una dimensión significa
*marginalizada*, convención OLAP estándar.

**Dos reglas de agregación de obligado cumplimiento**, implementadas en
`evaluation/aggregate.py`:

1. **Nunca se agrega una fila ya agregada.** Toda fila de `metrics` se calcula desde
   `forecasts`. MASE y sMAPE son cocientes: la media de cocientes no es el cociente de
   medias, y encadenar agregaciones produce números que no significan nada.
2. **El denominador de MASE se calcula por serie y por ventana, con el train de esa
   ventana**, como `mean(|y_t − y_{t−m}|)` con `m = spec.mase_season`. Usarlo global sería
   fuga. Se persiste como `metric="mase_denominator"` para que sea auditable desde la app,
   que es lo que convierte la afirmación "no hay fuga" en algo verificable por un tercero.

#### `dm_tests.parquet`

| Columna | Tipo | Descripción |
|---|---|---|
| `run_id` | `string` | |
| `model_a`, `model_b` | `string` | Comparación dirigida `a` vs `b` |
| `unique_id` | `string` \| `null` | `null` = agrupado |
| `h_step` | `int16` \| `null` | |
| `loss` | `string` | `abs` / `sq` / `pinball` |
| `stat`, `p_value` | `float64` | Estadístico DM y p-valor |
| `hac_lag` | `int16` | Retardo de la corrección HAC (`gap + h − 1`) |
| `hln_corrected` | `bool` | Corrección de muestra pequeña de Harvey-Leybourne-Newbold |
| `n_obs` | `int64` | |
| `n_comparisons` | `int32` | Comparaciones del run, para contextualizar la multiplicidad |

Solo se calcula sobre `stage="holdout"`. Con 12 modelos hay 66 pares y el p-valor pierde
sentido sin control de multiplicidad: la tabla incluye `n_comparisons` y la app muestra
además el **Model Confidence Set**, que es la herramienta correcta para "qué modelos no
puedo descartar", en lugar de 66 tests sueltos.

#### `anomaly_scores/detector_id=<k>/*.parquet`

| Columna | Tipo | Descripción |
|---|---|---|
| `unique_id` | `string` (dict) | |
| `ds` | `timestamp[ns]` | |
| `score` | `float32` | Mayor = más anómalo. Ordinal dentro de (detector, serie) |
| `scorable` | `bool` | `False` en calentamiento o donde `y` es `NaN` |
| `base_model_id` | `string` \| `null` | Modelo del que salieron los residuos, si aplica |

#### `anomaly_thresholds.parquet`

| Columna | Tipo | Descripción |
|---|---|---|
| `run_id`, `detector_id` | `string` | |
| `unique_id` | `string` \| `null` | `null` = umbral global del detector |
| `alpha` | `float32` | Rejilla precomputada, p. ej. 40 valores en `[0.001, 0.2]` |
| `threshold` | `float32` | Score a partir del cual se marca anomalía |

Precomputar la rejilla convierte el slider de α de la app en una búsqueda en tabla. Sin
esto habría que recalcular cuantiles conformales en caliente, que es cálculo — prohibido
por A5.

#### `datasets/dataset_id=<d>/anomaly_truth.parquet`

| Columna | Tipo | Descripción |
|---|---|---|
| `unique_id` | `string` | |
| `ds` | `timestamp[ns]` | |
| `is_anomaly` | `bool` | |
| `event_id` | `string` \| `null` | Agrupa puntos contiguos del mismo evento |
| `anomaly_type` | `string` \| `null` | `spike`, `level_shift`, `variance_shift`, `seasonal_phase`, `sensor_freeze` |
| `severity` | `float32` \| `null` | Magnitud de la inyección, en desviaciones típicas locales |
| `injection_seed` | `int32` \| `null` | Reproducibilidad de la inyección |

Vive a nivel de **dataset**, no de run: la verdad no depende del detector ni del modelo.
Que sea una tabla aparte es lo que permite evaluar detectores nuevos sin regenerar nada.

#### `anomaly_events.parquet`

| Columna | Tipo | Descripción |
|---|---|---|
| `run_id`, `detector_id`, `unique_id` | `string` | |
| `event_id` | `string` | Evento **detectado** |
| `alpha` | `float32` | α con el que se derivó |
| `start_ds`, `end_ds` | `timestamp[ns]` | Extremos inclusivos |
| `n_points` | `int32` | Puntos marcados |
| `peak_score` | `float32` | Máximo del score en el evento |
| `matched_truth_event_id` | `string` \| `null` | Evento real emparejado, si lo hay |
| `match_kind` | `string` | `hit` / `false_alarm` / `missed` |

Es la tabla que alimenta directamente la tabla de eventos de la app y la curva
precisión-recall por tipo de anomalía.

#### `explanations.parquet`

| Columna | Tipo | Descripción |
|---|---|---|
| `run_id`, `model_id` | `string` | |
| `unique_id` | `string` \| `null` | `null` = global |
| `kind` | `string` | `gain`, `split`, `shap`, `attention_variable`, `attention_temporal` |
| `feature` | `string` | Nombre de la feature o de la variable de entrada |
| `ds` | `timestamp[ns]` \| `null` | Solo para atención temporal |
| `value` | `float64` | |

### 7.5 Dos desnormalizaciones deliberadas

Ambas violan la tercera forma normal a propósito y ambas tienen la misma justificación:
la app no puede calcular (A5), y una unión de millones de filas en Streamlit es cálculo.

- **`y` dentro de `forecasts`.** Es el eje de casi todos los gráficos. Es inmutable para
  una versión dada del dataset, así que no hay riesgo de divergencia mientras el
  `dataset_id` sea versionado.
- **`cutoff` dentro de `forecasts`.** Evita unir con `windows` para colorear por origen,
  que es la interacción más frecuente de la página de forecast.

### 7.6 Artefactos demo vs artefactos completos

| | Ruta | Versionado en git | Contenido |
|---|---|---|---|
| Completos | `data/artifacts/` | No (gitignored) | Todas las series, todos los modelos, todas las ventanas |
| Demo | `reports/artifacts/` | **Sí** | Subconjunto curado: ~6 series, horizonte completo, todos los modelos |

`scripts/build_demo_artifacts.py` deriva el segundo del primero. El `.gitignore` ya
permite parquet bajo `reports/` y lo excluye en el resto del árbol, que es exactamente
esta política. La app resuelve la raíz con `CHRONOLAB_ARTIFACT_ROOT`, con
`reports/artifacts/` por defecto — de ahí que arranque sin red y sin datos descargados,
que es el criterio de calidad §6 del plan. **CI debe fallar si `reports/` supera 50 MB**;
sin ese guardarraíl, el repo engorda hasta romper el despliegue.

---

## 8. Fuga de información: dónde puede aparecer y qué la impide

Trece puntos. Para cada uno: dónde surge, la barrera y su **fuerza** según A3
(① ausencia física · ② constructor único · ③ aserción en el único camino · ④ lint ·
⑤ test). Las barreras de fuerza ④ y ⑤ están marcadas como tales con honestidad: son más
débiles, y donde solo hay eso, se dice.

| # | Punto de fuga | Barrera estructural | Fuerza |
|---|---|---|---|
| **L1** | **Partición no temporal** (split aleatorio, `train_test_split`) | `RollingOriginSplitter.split()` es el único emisor de particiones y las construye por aritmética sobre la rejilla. No existe API que acepte máscaras booleanas ni índices arbitrarios. `Window` es frozen y se autovalida | ② |
| **L2** | **Escalado o imputación ajustados con todo el dataset** | La arquitectura **no tiene etapa global de preprocesado**. `Panel` no expone `.scale()`, `.impute()` ni `.transform()`. Todo preprocesado dependiente de datos vive dentro de `Forecaster.fit(train)`, que solo recibe `ds <= cutoff` | ① |
| **L3** | **Features con ventana centrada o prospectiva** | `features/ops.py` no exporta ninguna primitiva prospectiva sobre columnas con `max_lead=0`; `lead()` sobre histórica lanza en construcción. Refuerzo: **test de estabilidad por prefijos** — para todo `t`, `features(panel[:t])` debe coincidir exactamente con `features(panel)[:t]`. Cualquier operación que mire adelante rompe esa igualdad, sea cual sea su forma | ① + ⑤ |
| **L4** | **Exógena futura que no era conocible** (reanálisis meteorológico como si fuese previsión) | `FutrProvider` con `Vintage` explícito; `RealizedFutrProvider` emite `PerfectForesightWarning`; el vintage entra en el `config_hash`, se persiste en `runs` y `aggregate` **rechaza** comparar filas de vintages distintos | ② + ③ |
| **L5** | **Tuning sobre las ventanas de reporte** | `BacktestPlan` expone `dev_windows` y `holdout_windows` por separado; la firma del optimizador acepta `Sequence[DevWindow]`, un tipo distinto de `Window`, así que pasarle ventanas de holdout no compila bajo mypy estricto. El leaderboard publica solo `stage="holdout"` | ① + ④ |
| **L6** | **Predecir instantes ya vistos** (contexto de Chronos que invade el test, off-by-one en el `gap`) | `FittedForecaster.cutoff` + comprobación `ds > cutoff` en `predict`, en el único camino de predicción. Intervalos semiabiertos `[start, end)` uniformes en todo el proyecto | ③ |
| **L7** | **Exógena histórica leída en el futuro** | `FutrFrame` contiene solo columnas `futr_exog`. No están omitidas por convenio: no existen en la estructura | ① |
| **L8** | **Feature derivada usada más allá de su disponibilidad** (`lag(y,24)` para predecir a 48 pasos sin recursión) | Álgebra de `max_lead` (§4.4): el motor filtra features por `max_lead >= L` en estrategia directa; la recursiva exige `recursive_only=True` y `supports_recursive` | ① + ③ |
| **L9** | **Calibración conformal solapando el tramo puntuado** | `FittedDetector.cutoff` y aserción `frame.start > cutoff` en `score`. `ScoringFrame` solo lo construye `artifacts.reader`, a partir de predicciones que ya son fuera de muestra | ② + ③ |
| **L10** | **Imputación de huecos por interpolación** (usa valores futuros) | I3 conserva los huecos como `NaN` explícito en el panel canónico; imputar es responsabilidad del modelo, dentro de `fit`. Test que prohíbe `bfill`, `backfill`, `interpolate(...)` sin `limit_direction="forward"` y `fillna(method="bfill")` fuera de `models/adapters/`. `ffill` sí se permite: mira hacia atrás | ① + ④ |
| **L11** | **DST: horas duplicadas y ausentes** | I2 (UTC ingenuo obligatorio, tz-aware = error duro). En UTC el problema no existe; la conversión ocurre una sola vez en `to_utc_naive()`. Fixtures de test con los cambios de hora reales de Europe/Madrid | ① + ⑤ |
| **L12** | **La app recalcula y se salta el arnés** | `app/` y `api/` tienen prohibido importar `models`, `evaluation` y `anomaly` (regla `banned-api` de ruff, §2.1). El extra `app` de `pyproject.toml` ni siquiera instala esas dependencias: el import fallaría en runtime | ④ + ① |
| **L13** | **Inyección de anomalías contaminando la calibración** | `injection.py` devuelve `(panel, truth)` y aplica las anomalías solo en la región de evaluación declarada; la calibración de detectores usa tramos anteriores al `cutoff` del detector, que L9 ya garantiza | ③ |

### 8.1 Los tres tests que hacen creíble todo lo anterior

El criterio de calidad del plan pide *"al menos un test que detectaría la fuga si la
introdujeses"*. Se implementan tres, en `tests/leakage/`:

**T1 — Estabilidad por prefijos.** Para una rejilla de cortes `t`, se verifica
`features(panel.slice(:t)).equals(features(panel).slice(:t))`. Es una propiedad general:
detecta cualquier operación prospectiva sin necesidad de enumerarlas. Se ejecuta con
`hypothesis` sobre paneles sintéticos.

**T2 — Canario, control positivo.** Se añade al panel una columna `_canary = y` declarada
como `futr_exog` y se lanza un backtest corto. Un modelo con exógenas futuras **debe**
bajar su MASE a prácticamente cero. Si no lo hace, el canal de exógenas futuras está roto
o desconectado — y todos los resultados con exógenas serían basura sin que nada lo delate.
Verifica que el cableado *funciona*.

**T3 — Canario, control negativo.** La misma columna `_canary = y`, ahora declarada como
`hist_exog`. El MASE **no** debe mejorar de forma apreciable, porque en el tramo futuro esa
columna no debe estar disponible. Si mejora, hay fuga por el canal histórico.

T2 y T3 juntos son la prueba que convierte "creo que no hay fuga" en "lo he medido en
ambos sentidos". Es el par que conviene enseñar en el README.

Complementos: propiedades del splitter con `hypothesis` (`cutoff < first_pred`,
`first_pred - cutoff == (gap+1)*freq`, ventanas disjuntas en train cuando corresponde),
fixtures de DST con los cambios reales de 2012-2024, y un test de coherencia que recalcula
tres métricas desde `forecasts` y las compara con `metrics.parquet`.

---

## 9. Decisiones de diseño

Índice normativo. Cada decisión con su justificación y, cuando la tiene, su alternativa
descartada. Las que ya se argumentaron arriba llevan referencia en vez de repetirse.

**D1 — `Panel = (df, spec)`; ningún `DataFrame` desnudo cruza módulos.**
El formato largo no declara roles, frecuencia ni huso: justo lo que produce fuga.
*Alternativa descartada:* convenciones de nombres (`temp_futr_`, `temp_hist_`). Es
*stringly-typed*, no lo verifica nadie y se rompe al primer renombrado. → §3.2, A2.

**D2 — `ds` en UTC ingenuo, con tz-aware como error duro.**
Elimina el DST por construcción y evita la fricción de tz-aware con Nixtla y parquet. El
error duro impide que convivan los dos convenios. *Alternativa descartada:* tz-aware UTC,
más seguro en teoría pero con conversiones implícitas en cada frontera de librería. → I2.

**D3 — Rejilla completa con `NaN` explícito.**
Convierte "falta un dato" —invisible— en un valor auditable, y garantiza que pivotar a
ancho es seguro. Coste: filas de más y adaptadores que deben tratar `NaN` según la
convención de cada librería. Ese trabajo es explícito y revisable, que es justamente lo
contrario de un bug silencioso. → I3.

**D4 — Motor de backtesting propio; Nixtla solo para modelos.** → §6.3.

**D5 — `fit` devuelve un `FittedForecaster` inmutable que conoce su `cutoff`.**
Un `Forecaster` mutable permite ajustar una vez y predecir en todas las ventanas sin que
nada lo impida. Con objetos nuevos por ventana, la reutilización es explícita y queda
registrada en `model_runs.refit`. Además habilita paralelizar por ventana sin estado
compartido. → §5.2.

**D6 — `h` es parámetro de `fit`, no de `predict`.**
`neuralforecast`, PatchTST y la estrategia directa lo necesitan en construcción; y
elimina la posibilidad de desajuste entre el `h` de entrenamiento y el de predicción.

**D7 — `FutrFrame` con constructor único (`FutrProvider`).** → §4.2.

**D8 — Vintage explícito para exógenas futuras, con tres proveedores.** → §4.3. Es la
decisión que más protege el resultado principal del proyecto.

**D9 — `max_lead` entero en lugar de futuro/histórico binario.** → §4.4. Sin esto, la
comparación recursiva vs directa (Fase 6) no es interpretable.

**D10 — Los cuantiles son la representación canónica; `lo`/`hi` se derivan.** → §7.3.

**D11 — Rejilla fija de siete cuantiles en columnas anchas.**
Alternativa descartada: cuantiles en formato largo, que multiplicaría por siete la tabla
más grande del proyecto para nada — la rejilla es fija por configuración de run. Nota de
honestidad: el CRPS calculado desde siete cuantiles es una **aproximación discreta**
(pinball promediado sobre la rejilla) y se nombra `crps_discrete` en `metrics`, no `crps`.
Llamarlo CRPS a secas sería sobrevender.

**D12 — Los detectores consumen `ScoringFrame`, construido solo desde artefactos.**
Hace estructuralmente imposible puntuar sobre predicciones in-sample, y fuerza a que
todos los detectores se evalúen sobre exactamente los mismos instantes. Los detectores
que no usan predicción (MatrixProfile) simplemente ignoran `y_hat`. → §5.3.

**D13 — Puntuar y umbralizar son operaciones separadas; el score es ordinal.** → §5.3.

**D14 — El dataset de evaluación es un subconjunto curado de series, no las 370.**
La malla de evaluación es `modelos × series × ventanas × refits` y es multiplicativa.
Prophet ajusta un modelo por serie: 370 series × 20 ventanas = 7.400 ajustes por run.
Decisión: `es_demand_h` con 6-10 series representativas (elegidas por estratificación
según entropía espectral y fuerza estacional, no a dedo) para el leaderboard completo, y
un run aparte de solo modelos baratos sobre el panel completo para demostrar que escala.
Es una decisión de alcance disfrazada de decisión técnica, y por eso está escrita.

**D15 — Baselines reimplementados en numpy además de vía `statsforecast`.**
`baselines.py` da una implementación independiente de `Naive` y `SeasonalNaive`, y un
test verifica que coincide con la de `statsforecast` en las mismas ventanas. Si el arnés
es el producto, el arnés necesita un patrón de referencia calculable a mano. También
protege frente a un cambio de convención de la librería entre versiones.

**D16 — `pandera` es el sistema de tipos en runtime; `adapters/` es la cuarentena de `Any`.**
mypy estricto no ve dentro de un `DataFrame`; `pandera` sí. Regla: las librerías de
modelado sin tipos se importan **solo** bajo `models/adapters/` y `anomaly/`, con
`ignore_missing_imports` limitado a esos módulos, y ningún `Any` sale de ahí: lo que sale
es un `DataFrame` validado contra su esquema.

**D17 — Reglas de agregación de métricas; MASE con denominador por ventana y persistido.**
→ §7.4. La auditabilidad del denominador es lo que hace verificable la ausencia de fuga.

**D18 — Artefactos demo versionados vs completos ignorados, con tope de tamaño en CI.**
→ §7.6.

**D19 — Los fallos ocupan una fila (`status`).** Un modelo que desaparece del leaderboard
por reventar es un leaderboard mentiroso. → A6.

**D20 — Extras de dependencias en `pyproject.toml`: `app`, `stats`, `ml`, `deep`,
`foundation`, `dev`.**
Streamlit Community Cloud instalando `torch`, `prophet` y `neuralforecast` es lento y
frágil, y no los necesita porque la app no calcula (A5). El extra `app` trae solo
`streamlit`, `plotly`, `pandas`, `pyarrow` y `pandera`. Efecto colateral valioso: si
alguien intenta violar L12, el import falla en el entorno desplegado.

**D21 — El `registry` construye `Forecaster`s a partir de un `PanelSpec`.**
`neuralforecast` necesita las listas de exógenas al construir la red, no al ajustar. Por
tanto la fábrica es `build(model_id, *, spec, seed, params) -> Forecaster` y la
construcción del objeto subyacente se difiere a `fit`, donde ya se conoce `h`. Es una
consecuencia poco obvia de D6 y conviene tenerla presente desde el principio, porque
rehacerla después toca todos los adaptadores.

**D22 — `ConformalWrapper` en `models/wrappers.py`, distinto del detector conformal.**
Convierte cualquier `Forecaster` puntual en probabilístico, calibrando sobre un corte
interno del propio train dentro de `fit`. Separarlo del detector de `anomaly/conformal.py`
evita confundir dos cosas distintas: producir intervalos (modelo) y decidir qué cae fuera
(detector).

**D23 — Un `run` cubre un dataset y un plan de backtesting.** → §7.1.

---

## 10. Discrepancias con `PLAN_PROYECTO.md`

Ocho, ordenadas por importancia. Las cuatro primeras cambian resultados; las demás son
de ingeniería.

**X1 — Open-Meteo *archive* no es una "covariable exógena futura conocida".**
*El plan* (§0, tabla de fuentes; §1, tabla de stack) la presenta así. Es reanálisis, es
decir, el valor revisado a posteriori: nadie lo tenía en el `cutoff`. Usarlo como exógena
futura infla sistemáticamente a los modelos que más dependen de la exógena —LightGBM,
TFT— frente a los baselines, que es exactamente la comparación que el proyecto quiere
hacer. **Es la fuga con más capacidad de invalidar la conclusión principal y no deja
ningún síntoma.** Propuesta: `FutrProvider` con vintage explícito (§4.3), Open-Meteo
*historical-forecast* donde exista cobertura, y previsión simulada calibrada donde no.

**X2 — Adoptar el formato de Nixtla no equivale a tener un contrato de datos.**
*El plan* (§1) presenta "adoptar el contrato de datos de Nixtla" como la decisión
arquitectónica clave. El formato largo es un buen formato de **transporte**, pero es mudo
sobre roles, frecuencia, huso y huecos. Propuesta: `PanelSpec` + `Panel` + los siete
invariantes (§3).

**X3 — No usar `cross_validation` de Nixtla como motor de backtesting.** → §6.3.
*El plan* (§1) lo lista como una de las ventajas de adoptar el formato. Es la ventaja que
hay que rechazar, precisamente porque el arnés es el producto.

**X4 — La distinción futuro/histórico es insuficiente; hace falta `max_lead`.**
*El plan* (Fase 4) pide "variables exógenas separadas en conocidas a futuro vs solo
históricas". Correcto pero incompleto: las **features derivadas** tienen disponibilidad
graduada, y es ahí donde se rompe la comparación recursiva vs directa que el propio plan
pide en la Fase 6. → §4.4.

**X5 — `reports/results/leaderboard.parquet` no basta.**
*El plan* (Fase 4) persiste un único fichero plano. La app necesita predicciones por
`(serie, modelo, ventana, paso)` con cuantiles, umbrales precomputados por α, verdad de
anomalías, eventos y costes; y necesita leerlos sin unir tablas grandes en caliente.
Propuesta: el esquema en estrella de §7. El leaderboard pasa a ser una vista sobre
`metrics.parquet`, no un fichero.

**X6 — La malla de evaluación es multiplicativa y hay que acotarla en el diseño, no al
final.** *El plan* no dimensiona el coste. Prophet ajusta por serie; `neuralforecast`
reajustar en cada ventana es prohibitivo. Propuesta: D14 (subconjunto curado) + política
de refit declarada y persistida + `refit_cost` en `ModelRequirements`.

**X7 — El árbol de módulos del plan mezcla familias de modelos con adaptadores.**
*El plan* propone `models/{baselines,statistical,ml,deep,foundation}.py`. La división útil
no es "qué tipo de modelo es" sino "qué librería hay que envolver": los ficheros por
familia acaban importando dos o tres librerías cada uno y la cuarentena de `Any` (D16)
deja de ser delimitable. Propuesta: `models/adapters/<librería>.py`, uno por backend, más
`registry.py`, `wrappers.py` y `baselines.py`. Cambios menores en la misma línea:
`base.py` → `protocols.py`; módulos nuevos `panel.py`, `types.py`, `artifacts/`,
`features/roles.py`, `data/align.py`, `data/futr.py`, `anomaly/injection.py`,
`evaluation/splitters.py`.

**X8 — "~2000 líneas sólidas" es optimista.**
*El plan* (§1) estima que adoptar Nixtla reduce el proyecto a ~2000 líneas. Solo el arnés
honesto —splitter, motor, métricas, agregación, artefactos, esquemas y los tests de
fuga— ronda las 2000. Con seis adaptadores, cuatro detectores, cinco páginas de app y la
capa de datos, la estimación realista está entre 5000 y 6000 líneas de código propio. No
cambia el plan; cambia el cronograma, y por eso está en R10.

Puntos del plan que asumo sin cambios y conviene dejar dicho: demanda eléctrica como
serie principal, cripto como contraste declarado, MASE como métrica principal,
Diebold-Mariano, rechazo del *point-adjusted* F1 en favor de VUS-PR y F1 por rangos,
`uv` como gestor, ruff/mypy/pytest, Streamlit sobre Gradio, y el control de versiones
manual con `CLAUDE.md` + `.claude/settings.json`.

---

## 11. Riesgos, por gravedad

Gravedad = impacto × probabilidad × dificultad de detección. Los tres primeros comparten
una propiedad que los hace peores que el resto: **fallan en silencio y con resultados
plausibles**.

### R1 — Presciencia en las exógenas futuras · gravedad: crítica

*Qué pasa:* se usa temperatura de reanálisis como si fuese previsión. LightGBM y TFT
baten a `SeasonalNaive` con holgura, el leaderboard se publica y la conclusión principal
del proyecto es falsa.
*Por qué es el peor:* no hay síntoma. Los números son bonitos y coherentes.
*Mitigación:* D8/§4.3 — vintage explícito, `PerfectForesightWarning`, vintage en el hash
de configuración y en `runs`, negativa a comparar vintages distintos.
*Señal temprana:* si la mejora de un modelo con exógenas sobre el mismo modelo sin ellas
supera el ~15-20 % de MASE en demanda eléctrica, sospechar del vintage antes de celebrar.
*Riesgo residual:* con `SIMULATED_FORECAST` el error sintético puede estar mal calibrado.
Se acota comparando los tres vintages sobre la serie de REE, donde existen los tres.

### R2 — Explosión combinatoria de la malla de evaluación · gravedad: alta

*Qué pasa:* `modelos × series × ventanas × refits`. Un run que debía tardar 20 minutos
tarda 14 horas, el ciclo de iteración muere y el proyecto se para en la Fase 6 o 7.
*Mitigación:* D14 (subconjunto curado), `refit_cost` con política por defecto
(`expensive` → un solo ajuste), paralelización por `(modelo, ventana)`, y un
**presupuesto de tiempo por run declarado en `conf/backtest.yaml`** que el motor estima
antes de empezar y sobre el que avisa si se excede.
*Señal temprana:* la primera ejecución de Prophet sobre el panel completo.

### R3 — DST, huecos y duplicados corrompiendo la estacionalidad · gravedad: alta

*Qué pasa:* la estacionalidad de 24 h se desalinea dos veces al año. Todo lo estacional
—que es casi todo— queda contaminado.
*Mitigación:* I2 + I3 + fixtures de DST reales. Es el riesgo mejor cubierto de la lista.
*Riesgo residual:* datos con horas locales mal etiquetadas en origen, que ninguna barrera
puede detectar. Contramedida: `data/quality.py` reporta densidad por hora del día y por
mes; una anomalía en esa tabla delata el problema.

### R4 — El ground truth de anomalías es sintético · gravedad: media-alta

*Qué pasa:* las conclusiones sobre qué detector captura qué dependen del generador de
anomalías, no del mundo. Con inyecciones demasiado fáciles todos los detectores parecen
excelentes; con un solo tipo, el ranking es un artefacto del tipo elegido.
*Mitigación:* cinco tipos con severidad parametrizada y barrido de severidad; la curva
precisión-recall se reporta **por tipo**, nunca agregada a un número; el README declara
explícitamente que la evaluación es sobre anomalías inyectadas.
*Riesgo residual:* irreducible sin etiquetas reales. Se mitiga con honestidad, no con
técnica: es material para la sección "limitaciones".

### R5 — Deriva del esquema de artefactos · gravedad: media

*Qué pasa:* la app lee runs antiguos con columnas que ya no significan lo mismo, o el
cron sobrescribe mientras Streamlit lee.
*Mitigación:* `SCHEMA_VERSION` en manifest, lector que **rechaza** versiones desconocidas,
escritura atómica con `manifest.json` al final.
*Señal temprana:* cualquier `KeyError` en la app al cambiar de run.

### R6 — Fragilidad del entorno · gravedad: media

*Qué pasa:* `prophet` requiere toolchain de compilación; `torch` pesa cientos de MB en CI;
las tres librerías Nixtla comparten `utilsforecast` y se desincronizan entre versiones;
`neuralforecast` arrastra `pytorch-lightning`.
*Mitigación:* `uv.lock` commiteado, extras por capa (D20), CI que instala solo los extras
que el job necesita, y matriz de CI con un job ligero (lint + tests unitarios, sin ML) que
es el que da feedback rápido.
*Señal temprana:* el primer `uv sync` en un runner limpio.

### R7 — Chronos: descarga, CPU y modo sin red · gravedad: media

*Qué pasa:* el modelo se descarga de Hugging Face; en CI o en la demo offline no hay red;
en CPU la latencia puede ser incómoda con contextos largos.
*Mitigación:* Chronos-Bolt (corre en CPU), caché del modelo en el runner, y —clave— la
app **no ejecuta Chronos**, solo lee sus predicciones ya persistidas (A5). El modo demo
nunca lo necesita.

### R8 — Límites de Streamlit Cloud y tamaño del repo · gravedad: media-baja

*Qué pasa:* los artefactos demo crecen, el repo engorda, el despliegue se ralentiza o
falla por memoria.
*Mitigación:* D18 + control de tamaño de `reports/` en CI + `@st.cache_data` con TTL sobre
las funciones de `artifacts.reader` + lectura con proyección de columnas y filtro por
partición (nunca `read_parquet` de un directorio entero).

### R9 — `mypy --strict` contra librerías sin tipos · gravedad: baja (fricción alta)

*Qué pasa:* `# type: ignore` se propaga y el tipado deja de significar nada.
*Mitigación:* D16 — la cuarentena está delimitada por módulo y la frontera real de tipos
en runtime es `pandera`.

### R10 — Alcance y cronograma · gravedad: baja (probabilidad alta)

*Qué pasa:* seis semanas para doce fases con una persona es optimista, sobre todo con la
estimación de líneas corregida en X8.
*Mitigación:* el propio plan ya da el orden de recorte correcto —fases 0-5 + 8 + 9 son un
proyecto excelente y el deep learning es lo primero que se recorta, no el backtesting—.
Esta arquitectura lo respeta: `models/adapters/` permite añadir backends sin tocar el
motor, así que las fases 7 y 12 son estrictamente aditivas.

### R11 — Selección múltiple de modelos · gravedad: baja (pero sutil)

*Qué pasa:* con 12 modelos hay 66 comparaciones por pares; a α=0.05 se esperan ~3
"significativas" por puro azar. Publicar "mi modelo bate al baseline con p<0.05" tras
haber probado doce es un abuso del test.
*Mitigación:* `n_comparisons` persistido en `dm_tests`, Model Confidence Set en la app
como respuesta principal, y separación dev/holdout (L5) para que la selección no se haga
sobre las ventanas reportadas.

---

## 12. Orden de implementación sugerido

Deriva del árbol de dependencias, no de las fases del plan. Cada hito deja el repo verde.

| Hito | Contenido | Por qué en este orden |
|---|---|---|
| **H1** | `types`, `errors`, `panel`, `data/schemas`, `data/align`, `data/sources/synthetic` | Todo lo demás depende del contrato de datos. La fuente sintética permite testear sin red desde el primer día |
| **H2** | `evaluation/splitters` + tests de propiedad + `features/ops` y `features/roles` + T1 | El arnés antes que los modelos (A1). T1 debe existir antes de la primera feature real |
| **H3** | `artifacts/` completo + `models/baselines` + `models/protocols` | Con baselines en numpy y artefactos ya se puede cerrar el bucle end-to-end |
| **H4** | `evaluation/backtest` + `metrics` + `aggregate` + T2/T3 | Primer leaderboard real, aunque solo con baselines. **Aquí el proyecto ya es defendible** |
| **H5** | `data/sources/*` reales + `data/futr` + `data/quality` | Datos de verdad sobre un arnés ya probado, no al revés |
| **H6** | `models/adapters/*` uno a uno, en orden statsforecast → mlforecast → prophet → neuralforecast → torch_lstm → chronos | Cada adaptador es aditivo; el motor no cambia |
| **H7** | `anomaly/*` + `evaluation/anomaly_metrics` + `injection` | Depende de artefactos de forecast ya existentes (D12) |
| **H8** | `viz`, `app`, `api`, workflows de CI y de refresco | Lo último: consume, no produce |

Nota de método: **H4 antes que H5** es contraintuitivo pero deliberado. Un leaderboard de
baselines sobre datos sintéticos con verdad conocida es la única forma de saber que el
arnés mide lo que dice medir antes de que los datos reales introduzcan ruido que enmascare
sus errores.

---

## 13. Fuera del alcance de este documento

Se decidirán cuando toque, y no bloquean la implementación: elección exacta de
hiperparámetros y espacios de búsqueda de Optuna; diseño visual de la app; contenido del
README; definición formal de las métricas (irá en `docs/METHODOLOGY.md`); Dockerfile;
model cards. También queda pendiente, como deuda declarada, la estructura para
reconciliación jerárquica (§3.2, punto 3), que el formato largo no soporta por sí solo.

---

## 14. Aprobación

Este documento propone, respecto al plan original, cuatro cambios que afectan a los
resultados —vintage de exógenas (X1), contrato tipado sobre el formato largo (X2), motor
de backtesting propio (X3) y `max_lead` en lugar de futuro/histórico binario (X4)— y
cuatro de ingeniería (X5-X8).

**¿Apruebas este diseño?** En concreto, y si algo va a cambiar, es más barato ahora:

1. **X3, motor de backtesting propio en lugar de `cross_validation` de Nixtla.** Es el
   cambio con más coste de implementación (~400 líneas) y el que más difícil sería
   revertir después. ¿Lo asumimos?
2. **X1 + D8, vintage de exógenas.** ¿Empezamos con `SIMULATED_FORECAST` como defecto
   para UCI y `ARCHIVED_FORECAST` para REE, o prefieres arrancar con `REALIZED` etiquetado
   como presciencia perfecta y añadir los otros dos más adelante?
3. **D14, subconjunto curado de 6-10 series** en lugar de las 370 de UCI para el
   leaderboard completo. ¿Te parece bien el número?
4. **D20, extras de dependencias** con una app que no instala `torch` ni `prophet`.
   ¿Confirmas que el despliegue objetivo es Streamlit Community Cloud?

No implementaré nada —ni el árbol de módulos, ni `pyproject.toml`, ni un solo módulo—
hasta que confirmes.








