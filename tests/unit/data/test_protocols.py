"""Conformidad estructural con el protocolo `DataSource`."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from chronolab.data.protocols import DataSource, SourceSpec
from chronolab.types import Role, SeriesId


class DummySource:
    """Fuente minima que satisface el protocolo sin heredar de el."""

    def __init__(self, spec: SourceSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    def fetch(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ids: Sequence[SeriesId] | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["unique_id", "ds", *self._spec.value_columns])


def _spec(**overrides: object) -> SourceSpec:
    kwargs: dict[str, object] = {
        "source_id": "dummy",
        "role": Role.TARGET,
        "value_columns": ("y",),
        "freq": "h",
    }
    kwargs.update(overrides)
    return SourceSpec(**kwargs)  # type: ignore[arg-type]


class TestConformidad:
    def test_una_implementacion_estructural_satisface_el_protocolo(self) -> None:
        # El protocolo se satisface por forma, no por herencia: eso permite
        # envolver librerias externas sin tocarlas.
        assert isinstance(DummySource(_spec()), DataSource)

    def test_un_objeto_sin_fetch_no_lo_satisface(self) -> None:
        assert not isinstance(object(), DataSource)


class TestSourceSpec:
    def test_es_inmutable(self) -> None:
        spec = _spec()
        try:
            spec.source_id = "otro"  # type: ignore[misc]
        except AttributeError:
            return
        raise AssertionError("SourceSpec deberia ser inmutable")

    def test_no_es_vintage_aware_por_defecto(self) -> None:
        # Por defecto una fuente no sabe responder "que se sabia en `as_of`", y
        # pedirselo debe ser un error y no un parametro ignorado.
        assert _spec().vintage_aware is False

    def test_una_fuente_tiene_un_unico_rol(self) -> None:
        # Forzarlo evita la abstraccion que se filtra: una fuente meteorologica
        # no devuelve una columna `y` que luego haya que renombrar.
        assert _spec(role=Role.FUTR_EXOG, value_columns=("temp_c",)).role is Role.FUTR_EXOG

    def test_la_zona_nativa_por_defecto_es_utc(self) -> None:
        assert _spec().native_tz == "UTC"
