.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-fast coverage app clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Instala el entorno completo (core + ml + deep + app + dev)
	uv sync --all-extras

lint:  ## Lint y comprobacion de formato con ruff
	uv run ruff check .
	uv run ruff format --check .

format:  ## Aplica formato y arregla lo autoarreglable
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## mypy en modo estricto sobre src/
	uv run mypy

test:  ## Suite completa de tests, con cobertura en terminal
	uv run pytest --cov --cov-report=term-missing

test-fast:  ## Tests rapidos (excluye los marcados slow o network), con cobertura
	uv run pytest -m "not slow and not network" --cov --cov-report=term-missing

coverage:  ## Cobertura completa con informe HTML navegable en htmlcov/index.html
	uv run pytest --cov --cov-report=term-missing --cov-report=html
	@echo "Informe HTML en htmlcov/index.html"

app:  ## Levanta la aplicacion Streamlit
	uv run --extra app streamlit run src/chronolab/app/main.py

clean:  ## Borra cachés y artefactos de build
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
