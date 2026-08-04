# chronolab

Chronolab es una plataforma de forecasting y detección de anomalías en series temporales.
Provee pipelines de datos, modelos de predicción y una app para explorar resultados.
Está pensada para series temporales multivariadas en entornos productivos.

## Reglas de control de versiones — INNEGOCIABLES

Nunca ejecutes los siguientes comandos: `git add`, `git commit`, `git push`, `git merge`,
`git rebase`, `git reset`, `git checkout -b`, ni ningún comando `gh`. El humano gestiona
todo el control de versiones.

Sí puedes usar: `git status`, `git diff`, `git log`.

## Al terminar cada tarea

Al finalizar cada tarea, imprime:

1. La lista de archivos creados, modificados y eliminados.
2. Un mensaje de commit sugerido en formato [Conventional Commits](https://www.conventionalcommits.org/).

## Convenciones de código

- Python 3.12
- Layout `src/`
- `ruff` para lint y formato
- `mypy` en modo estricto sobre `src/`
- Type hints obligatorios en funciones públicas
- Docstrings estilo NumPy

## Comandos del proyecto

- `make lint` — lint y formato con ruff
- `make test` — suite de tests
- `make app` — levanta la aplicación
