# chronolab

Plataforma de forecasting y detección de anomalías en series temporales.

> **Estado: andamiaje.** El árbol de módulos y los contratos centrales están
> definidos; los algoritmos todavía no. La referencia normativa del diseño es
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), y el plan de ejecución está en
> [`docs/PLAN_PROYECTO.md`](docs/PLAN_PROYECTO.md).

El diferenciador del proyecto no son los modelos —son commodities— sino el rigor
del marco de evaluación: backtesting de origen rodante con barreras estructurales
contra la fuga de información temporal, métricas escala-independientes y
evaluación de anomalías por eventos.

## Puesta en marcha

```bash
uv sync                 # core + dev: suficiente para lint, typecheck y tests
uv sync --all-extras    # añade los modelos (ml, deep) y la app
```

## Comandos

| Comando | Qué hace |
|---|---|
| `make install` | Instala el entorno completo |
| `make lint` | Lint y comprobación de formato con ruff |
| `make format` | Aplica formato y arregla lo autoarreglable |
| `make typecheck` | mypy en modo estricto sobre `src/` |
| `make test` | Suite completa |
| `make test-fast` | Excluye los tests marcados `slow` y `network` |
| `make app` | Levanta la aplicación Streamlit |
| `make clean` | Borra cachés |

## Hallazgo: zero-shot frente a modelos ajustados

`chronolab.models.foundation` (adaptador en
[`models/adapters/chronos.py`](src/chronolab/models/adapters/chronos.py)) envuelve
**Chronos-Bolt-small** (47.7M parámetros, CPU, `amazon/chronos-bolt-small`) sin
ningún ajuste local: `fit` solo recorta el contexto de cada serie y fija el
cutoff. El leaderboard lo marca con `training_regime="zero-shot"` frente a
`"fitted"` en el resto.

Un backtest de origen rodante (6 ventanas, 3 de holdout, `h=24`, panel horario
sintético de 3 series con estacionalidad diaria y semanal —
`tests/fixtures/synthetic.make_hourly_panel`, semilla fija, reajuste por
ventana) da esto en el agregado de holdout:

| Modelo | Régimen | MASE | RMSE | sMAPE | `fit_seconds` (total, 6 ventanas) |
|---|---|--:|--:|--:|--:|
| MSTL (STL diario+semanal · ARIMA) | fitted | **0.296** | 0.862 | 0.618 | 34.5 s |
| **Chronos-Bolt-small** | **zero-shot** | **0.313** | 0.892 | 0.653 | **0.011 s** |
| Naive estacional (24 h) | fitted | 0.902 | 2.488 | 1.855 | 0.007 s |
| AutoETS | fitted | 1.066 | 3.586 | 2.206 | 60.6 s |

**Lectura:** un modelo que nunca ha visto estas series queda a un 6 % de MASE
del mejor modelo ajustado del panel (MSTL, que además es el único de los tres
modelos clásicos que modela explícitamente la doble estacionalidad diaria y
semanal) y **supera claramente** tanto al naive estacional como a AutoETS —sin
ajustar nada, en milisegundos de `fit` frente a decenas de segundos—. Esto
confirma lo que reporta la literatura de modelos fundacionales de series
temporales (Chronos, TimesFM, Moirai): en series con estructura estacional
razonablemente regular, el zero-shot ya no es solo un baseline curioso, compite
de tú a tú con el ajuste clásico. No se esconde: se reporta tal cual.

Con matices que hay que leer junto al número:

- **Panel sintético, no productivo.** Es el generador de series de la propia
  suite de tests (`daily_amp`/`weekly_amp` conocidos, ruido gaussiano bajo). Es
  un caso favorable para cualquier modelo con buen prior estacional —Chronos
  incluido—, y no sustituye al backtest sobre UCI/REE del hito.
- **Coste de arranque no incluido en `fit_seconds`.** La descarga y carga de
  los pesos (~20 s la primera vez, cacheados después en disco por
  `huggingface_hub` y en memoria por proceso) es un coste fijo que no depende
  del tamaño del panel ni del número de ventanas; con paneles grandes se
  amortiza, con backtests de una sola ventana no.
- **Cobertura de colas más estrecha de lo ideal.** `predict_quantiles` clampa
  los cuantiles `0.025`/`0.975` de la rejilla canónica al límite nativo del
  modelo (`0.1`/`0.9`) en vez de extrapolar (ver el docstring del adaptador):
  la cola sale más estrecha que con un modelo calibrado a esos niveles.
- **MSTL y AutoETS no dejaron métricas probabilísticas en esta corrida**
  (`pinball_mean`/`coverage_50` a `NaN`) mientras que Chronos sí las produjo
  para los siete cuantiles pedidos; no se ha investigado la causa —parece un
  efecto de `ConformalIntervals` con `refit_every=1`, no del adaptador de
  Chronos— y se deja anotado en vez de callado.

## Control de versiones

Los commits los gestiona una persona, no el asistente. Las reglas están en
[`CLAUDE.md`](CLAUDE.md) y se hacen cumplir desde `.claude/settings.json`.

## Licencia

MIT.
