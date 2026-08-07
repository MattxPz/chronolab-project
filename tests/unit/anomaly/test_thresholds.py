"""Umbralizador conformal: el umbral se conoce en forma cerrada."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from chronolab.anomaly.conformal import ALPHA_GRID
from chronolab.anomaly.protocols import FittedThresholder, Thresholder
from chronolab.anomaly.thresholds import THRESHOLD_COLUMNS, ConformalThresholder


def _scores(values: list[float], *, scorable: list[bool] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": "s00",
            "ds": pd.date_range("2023-01-01", periods=len(values), freq="h"),
            "score": values,
            "scorable": [True] * len(values) if scorable is None else scorable,
        }
    )


class TestConformidad:
    def test_satisface_los_protocolos(self) -> None:
        thresholder = ConformalThresholder()
        assert isinstance(thresholder, Thresholder)
        assert isinstance(thresholder.fit(_scores([1.0, 2.0])), FittedThresholder)


class TestUmbral:
    @pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1, 0.2])
    def test_el_umbral_es_menos_log10_de_alfa(self, alpha: float) -> None:
        # Toda la calibracion vive dentro del score, asi que aqui no hay nada que
        # estimar. Estimarlo empiricamente anadiria error a un numero exacto.
        fitted = ConformalThresholder().fit(_scores([3.0]))
        row = fitted.threshold(alpha)
        assert len(row) == 1
        assert float(row.loc[0, "threshold"]) == pytest.approx(-math.log10(alpha), rel=1e-6)

    def test_el_umbral_es_global(self) -> None:
        # `unique_id` a nulo: la calibracion condicional por hora y adelanto ya
        # esta dentro del score, no en el umbral.
        row = ConformalThresholder().fit(_scores([3.0])).threshold(0.05)
        assert row.loc[0, "unique_id"] is None
        assert list(row.columns) == list(THRESHOLD_COLUMNS)

    def test_la_tabla_cubre_la_rejilla_completa(self) -> None:
        table = ConformalThresholder().fit(_scores([3.0])).table()
        assert len(table) == len(ALPHA_GRID)
        assert table["alpha"].is_monotonic_increasing
        assert table["threshold"].is_monotonic_decreasing

    def test_un_alfa_por_debajo_de_la_resolucion_se_marca_inalcanzable(self) -> None:
        # Un p-valor conformal no baja de 1/(n+1): por debajo de cierto alfa no
        # hay cola observada, y decirlo es preferible a devolver un umbral que
        # nadie podra cruzar nunca sin que nada lo advierta.
        fitted = ConformalThresholder().fit(_scores([1.0, 1.5, 2.0]))
        table = fitted.table().set_index("alpha")
        assert bool(table.loc[0.2, "reachable"])
        assert not bool(table.loc[0.001, "reachable"])

    def test_los_puntos_no_puntuables_no_cuentan_para_la_resolucion(self) -> None:
        fitted = ConformalThresholder().fit(
            _scores([1.0, 99.0], scorable=[True, False]),
        )
        assert fitted.score_ceiling == pytest.approx(1.0)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
    def test_un_alfa_fuera_de_rango_falla(self, alpha: float) -> None:
        fitted = ConformalThresholder().fit(_scores([1.0]))
        with pytest.raises(ValueError):
            fitted.threshold(alpha)

    def test_una_rejilla_incoherente_falla_al_construirse(self) -> None:
        with pytest.raises(ValueError):
            ConformalThresholder(alpha_grid=(0.2, 0.1))
        with pytest.raises(ValueError):
            ConformalThresholder(alpha_grid=())

    def test_calibrar_sin_las_columnas_del_contrato_falla(self) -> None:
        with pytest.raises(ValueError, match="scorable"):
            ConformalThresholder().fit(pd.DataFrame({"score": [1.0]}))
