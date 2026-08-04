"""Protocolos de prediccion: cuantiles canonicos, requisitos y conformidad."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest

from chronolab.models.protocols import (
    QUANTILES,
    FittedForecaster,
    Forecaster,
    ModelRequirements,
    quantile_column,
)
from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId


class DummyFitted:
    """Modelo ajustado minimo que satisface el protocolo."""

    def __init__(self, cutoff: pd.Timestamp, h: int) -> None:
        self._cutoff = cutoff
        self._h = h

    @property
    def model_id(self) -> ModelId:
        return ModelId("dummy")

    @property
    def cutoff(self) -> pd.Timestamp:
        return self._cutoff

    @property
    def h(self) -> int:
        return self._h

    @property
    def fit_seconds(self) -> float:
        return 0.0

    @property
    def n_params(self) -> int | None:
        return None

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])


class DummyForecaster:
    """Configuracion minima que satisface el protocolo."""

    @property
    def model_id(self) -> ModelId:
        return ModelId("dummy")

    @property
    def requires(self) -> ModelRequirements:
        return ModelRequirements()

    def fit(self, train: Panel, *, h: int) -> DummyFitted:
        return DummyFitted(cutoff=train.df["ds"].max(), h=h)


class TestConformidad:
    def test_forecaster_estructural(self) -> None:
        assert isinstance(DummyForecaster(), Forecaster)

    def test_fitted_forecaster_estructural(self) -> None:
        assert isinstance(DummyFitted(pd.Timestamp("2023-01-01"), 24), FittedForecaster)

    def test_fit_devuelve_un_objeto_nuevo_con_su_propio_cutoff(self, hourly_panel: Panel) -> None:
        # `fit` no muta `self`: un `Forecaster` es configuracion y un
        # `FittedForecaster` es configuracion mas frontera de informacion.
        forecaster = DummyForecaster()
        fitted = forecaster.fit(hourly_panel, h=24)
        assert fitted.cutoff == hourly_panel.df["ds"].max()
        assert fitted.h == 24
        assert not hasattr(forecaster, "cutoff")

    def test_chronos_encaja_sin_entrenar(self, hourly_panel: Panel) -> None:
        # El caso degenerado que valida el diseno: un modelo zero-shot solo fija
        # la frontera de informacion, que es lo unico que comparten todos.
        fitted = DummyForecaster().fit(hourly_panel, h=24)
        assert fitted.fit_seconds == 0.0
        assert fitted.n_params is None


class TestCuantiles:
    def test_la_rejilla_canonica_es_creciente_y_esta_en_el_abierto(self) -> None:
        assert list(QUANTILES) == sorted(QUANTILES)
        assert all(0.0 < q < 1.0 for q in QUANTILES)

    def test_la_rejilla_incluye_la_mediana(self) -> None:
        assert 0.5 in QUANTILES

    @pytest.mark.parametrize(
        ("quantile", "expected"),
        [(0.025, "q_0250"), (0.1, "q_1000"), (0.5, "q_5000"), (0.975, "q_9750")],
    )
    def test_nombre_de_columna(self, quantile: float, expected: str) -> None:
        assert quantile_column(quantile) == expected

    def test_el_orden_lexicografico_coincide_con_el_numerico(self) -> None:
        # Es el motivo de rellenar a cuatro digitos: asi las columnas de un
        # parquet se ordenan solas de forma util.
        names = [quantile_column(q) for q in QUANTILES]
        assert names == sorted(names)

    @pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.5])
    def test_rechaza_cuantiles_fuera_del_abierto(self, quantile: float) -> None:
        with pytest.raises(ValueError, match="fuera de"):
            quantile_column(quantile)


class TestModelRequirements:
    def test_los_valores_por_defecto_son_los_conservadores(self) -> None:
        # Un modelo que no declara nada no recibe exogenas futuras ni se le
        # piden cuantiles: si algo se le pasa, es porque lo pidio.
        requirements = ModelRequirements()
        assert requirements.needs_futr_exog is False
        assert requirements.supports_quantiles is False
        assert requirements.supports_recursive is False
        assert requirements.is_zero_shot is False
        assert requirements.refit_cost == "cheap"

    def test_es_inmutable(self) -> None:
        with pytest.raises(AttributeError):
            ModelRequirements().needs_futr_exog = True  # type: ignore[misc]
