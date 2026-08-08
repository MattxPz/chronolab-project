"""Envoltorio tipado de `st.cache_data`.

`streamlit` no se instala en el job "core + dev" de CI que corre `mypy`
(solo lo trae el extra `app`, y ese job lo evita a proposito para quedarse
por debajo del minuto de instalacion): `pyproject.toml` declara
`ignore_missing_imports` para el modulo, lo que lo resuelve como `Any`.
Decorar directamente con ``@st.cache_data(...)`` bajo ``Any`` dispara
`disallow_untyped_decorators` en cada funcion cacheada de `state.py` y
`palette.py`. Esta funcion le da a `st.cache_data` la firma que streamlit no
publica para mypy, una sola vez, en vez de repetir ``# type: ignore`` en
veinte sitios.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import streamlit as st

__all__ = ["cache_data"]

_F = TypeVar("_F", bound=Callable[..., Any])


def cache_data(**kwargs: Any) -> Callable[[_F], _F]:
    """`st.cache_data(**kwargs)`, con una firma que mypy puede verificar.

    Parameters
    ----------
    **kwargs
        Se reenvian tal cual a ``st.cache_data`` (p. ej. ``show_spinner``).

    Returns
    -------
    Callable
        El mismo decorador que devuelve ``st.cache_data``, forzado al tipo
        ``Callable[[F], F]``: preserva la firma de la funcion que envuelve.
    """
    return cast(Callable[[_F], _F], st.cache_data(**kwargs))
