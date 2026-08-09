# syntax=docker/dockerfile:1
#
# Imagen unica para `chronolab.app` (Streamlit) y `chronolab.api.service`
# (FastAPI): las dos son de solo lectura de artefactos (docs/ARCHITECTURE.md
# A5 y §2.1) y comparten exactamente las mismas dependencias -por eso un solo
# Dockerfile y `docker-compose.yml` decide cual de las dos levantar por
# servicio, con `command:`. Ninguna arrastra `torch`, `prophet` ni
# `neuralforecast`: los extras `ml`/`deep` (el entrenamiento y el backtest)
# no entran en esta imagen a proposito.
#
# `scripts/refresh_data.py` -que si necesita `--extra ml`- no corre dentro de
# este contenedor: lo lanza `.github/workflows/refresh-data.yml` o, en local,
# `uv run --extra ml python scripts/refresh_data.py` desde el host. Ver
# docker-compose.yml para como la API lee lo que ese script escribe.

############################################################
# Etapa 1: build con uv
############################################################
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_EDITABLE=1

WORKDIR /src

# Capa de dependencias sola, cacheable con independencia del codigo: tocar un
# modulo de src/ no invalida esta capa mientras uv.lock no cambie.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-default-groups --extra app --extra api

# El paquete en si. README.md entra porque `project.readme` de pyproject.toml
# lo declara y hatchling lo exige para construir el wheel.
COPY src/ src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --extra app --extra api

############################################################
# Etapa 2: runtime slim, sin uv y sin usuario root
############################################################
FROM python:3.12-slim-bookworm AS runtime

# Nada en este contenedor necesita privilegios: la unica escritura posible es
# `data/artifacts/live/` (montada desde fuera, ver docker-compose.yml), y es
# de este usuario, no de root.
RUN groupadd --gid 1000 chronolab \
    && useradd --uid 1000 --gid chronolab --create-home --shell /usr/sbin/nologin chronolab

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHRONOLAB_RESULTS_DIR=/app/reports/results \
    CHRONOLAB_FIGURES_DIR=/app/reports/figures \
    CHRONOLAB_LIVE_DIR=/app/data/artifacts/live

WORKDIR /app

COPY --from=builder --chown=chronolab:chronolab /opt/venv /opt/venv
# `src/` se copia ademas del venv (que ya trae `chronolab` instalado, no
# editable) unicamente para que `streamlit run src/chronolab/app/main.py`
# funcione con la misma ruta que `make app`: streamlit necesita un fichero,
# no un modulo importable.
COPY --from=builder --chown=chronolab:chronolab /src/src /app/src
# reports/ es el subconjunto demo versionado (docs/ARCHITECTURE.md §2): lo
# necesita `chronolab.app` en modo demo, y `chronolab.api.service` si algun
# dia se apunta a `results_dir` en vez de a `live_dir`.
COPY --chown=chronolab:chronolab reports/ /app/reports/

# `data/artifacts/live/` lo escribe scripts/refresh_data.py fuera de este
# contenedor y lo lee `chronolab.api.service`. Se crea vacio y con el
# propietario correcto para que un bind mount ahi no herede permisos de root
# (ver el volumen del servicio `api` en docker-compose.yml).
RUN mkdir -p /app/data/artifacts/live && chown -R chronolab:chronolab /app/data

USER chronolab

EXPOSE 8000 8501

# Por defecto levanta la API; `docker-compose.yml` sobrescribe `command` para
# el servicio `app` con `streamlit run`.
CMD ["uvicorn", "chronolab.api.service:app", "--host", "0.0.0.0", "--port", "8000"]
