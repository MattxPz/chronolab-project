"""Resolucion de rutas de artefactos y ajustes de la app.

Alcance actual: solo lo que necesita `chronolab.app` para localizar
``reports/results/`` y ``reports/figures/`` sin depender de que el proceso se
lance desde la raiz del repositorio, mas el interruptor de modo demo. El
resto del alcance original de este modulo -modelos pydantic de
``conf/*.yaml`` y el hash canonico de configuracion de un run (A4)- todavia no
existe: ``conf/`` no se ha creado y ningun run se persiste con `run_id` fuera
de memoria. Se documenta aqui para que quien lo implemente no lo redescubra.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["AppSettings", "figures_dir", "get_settings", "live_dir", "results_dir"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
"""``src/chronolab/config.py`` -> ``src/chronolab`` -> ``src`` -> raiz del repo."""


class AppSettings(BaseSettings):
    """Ajustes de la app, sobrescribibles con variables de entorno ``CHRONOLAB_*``.

    Parameters
    ----------
    results_dir
        Directorio de artefactos precomputados (``*.parquet``) que lee
        `chronolab.artifacts.reader`. La app nunca escribe aqui.
    figures_dir
        Directorio de figuras estaticas (PNG) precomputadas por las notebooks
        y los scripts de ``scripts/``, para el material que aun no tiene una
        version interactiva (docs/ARCHITECTURE.md A5: la app no las genera).
    live_dir
        Directorio de artefactos de la capa de casi tiempo real, escritos por
        ``scripts/refresh_data.py`` y leidos por `chronolab.api.service`. A
        diferencia de `results_dir` -el subconjunto demo, versionado- este es
        un run completo que se sobrescribe cada refresco: vive bajo ``data/``
        (docs/ARCHITECTURE.md §2, "artifacts (runs completos, gitignored)") y
        `.gitignore` lo excluye con ``data/**``.
    demo_mode
        Si ``True``, la cabecera de la app avisa de que los artefactos son el
        subconjunto pequeno versionado en el repo, no un run completo.
    """

    model_config = SettingsConfigDict(env_prefix="CHRONOLAB_", frozen=True)

    results_dir: Path = _REPO_ROOT / "reports" / "results"
    figures_dir: Path = _REPO_ROOT / "reports" / "figures"
    live_dir: Path = _REPO_ROOT / "data" / "artifacts" / "live"
    demo_mode: bool = True


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Ajustes efectivos del proceso, resueltos una sola vez.

    Returns
    -------
    AppSettings
        Instancia cacheada: leer variables de entorno en cada acceso no
        aporta nada dentro de la vida de un proceso de Streamlit y rompería
        el presupuesto de arranque en frio.
    """
    return AppSettings()


def results_dir() -> Path:
    """Directorio de artefactos que consume la app.

    Returns
    -------
    pathlib.Path
    """
    return get_settings().results_dir


def figures_dir() -> Path:
    """Directorio de figuras estaticas precomputadas (PNG).

    Returns
    -------
    pathlib.Path
    """
    return get_settings().figures_dir


def live_dir() -> Path:
    """Directorio de artefactos de casi tiempo real que escribe/lee el refresco.

    Returns
    -------
    pathlib.Path
    """
    return get_settings().live_dir
