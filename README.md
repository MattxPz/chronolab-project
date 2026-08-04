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

## Control de versiones

Los commits los gestiona una persona, no el asistente. Las reglas están en
[`CLAUDE.md`](CLAUDE.md) y se hacen cumplir desde `.claude/settings.json`.

## Licencia

MIT.
