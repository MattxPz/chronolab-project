"""Esquemas pandera de tramas crudas: tipos, monotonia por serie, duplicados y rango."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from chronolab.data.schemas import (
    binance_schema,
    build_raw_schema,
    open_meteo_schema,
    ree_demand_schema,
    uci_electricity_schema,
)


def _frame(**overrides: object) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b"],
            "ds": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 00:00"]),
            "y": [1.0, 2.0, 3.0],
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return base


class TestBuildRawSchema:
    def test_acepta_una_trama_valida(self) -> None:
        schema = build_raw_schema({"y": (0.0, 100.0)})
        schema.validate(_frame(), lazy=True)

    def test_rechaza_ds_no_creciente_dentro_de_una_serie(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.to_datetime(["2024-01-01 01:00", "2024-01-01 00:00"]),
                "y": [1.0, 2.0],
            }
        )
        schema = build_raw_schema({"y": (0.0, 100.0)})
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(frame, lazy=True)

    def test_rechaza_ds_duplicado_dentro_de_una_serie(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00"]),
                "y": [1.0, 2.0],
            }
        )
        schema = build_raw_schema({"y": (0.0, 100.0)})
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(frame, lazy=True)

    def test_series_distintas_pueden_compartir_marca_de_tiempo(self) -> None:
        # Monotonia es por serie, no global: 'a' y 'b' comparten un ds sin que
        # eso sea un problema.
        schema = build_raw_schema({"y": (0.0, 100.0)})
        schema.validate(_frame(), lazy=True)

    def test_rechaza_valores_fuera_de_rango(self) -> None:
        schema = build_raw_schema({"y": (0.0, 2.0)})
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(_frame(), lazy=True)

    def test_admite_nan_en_columnas_de_valor(self) -> None:
        # Un hueco de origen es un NaN legitimo, no un error de esquema.
        schema = build_raw_schema({"y": (0.0, 100.0)})
        frame = _frame(y=[1.0, float("nan"), 3.0])
        schema.validate(frame, lazy=True)

    def test_rechaza_columnas_extra_no_declaradas(self) -> None:
        schema = build_raw_schema({"y": (0.0, 100.0)})
        frame = _frame(unexpected=[1, 2, 3])
        with pytest.raises(pandera.errors.SchemaErrors):
            schema.validate(frame, lazy=True)

    def test_coerciona_valores_numericos_en_texto(self) -> None:
        schema = build_raw_schema({"y": (0.0, 100.0)})
        frame = _frame(y=["1.0", "2.0", "3.0"])
        validated = schema.validate(frame, lazy=True)
        assert validated["y"].dtype == "float64"

    def test_acepta_una_trama_vacia(self) -> None:
        # Una consulta valida sin filas (por ejemplo, un rango de fechas sin
        # datos) es vacuamente monotona: no hay ningun par que la contradiga.
        # Regresion: `groupby(...).apply(...)` sobre cero grupos hacia
        # reventar `.all()` con un TypeError ajeno al problema real.
        empty = pd.DataFrame(
            {
                "unique_id": pd.Series(dtype="object"),
                "ds": pd.Series(dtype="datetime64[ns]"),
                "y": pd.Series(dtype="float64"),
            }
        )
        schema = build_raw_schema({"y": (0.0, 100.0)})
        validated = schema.validate(empty, lazy=True)
        assert len(validated) == 0


class TestEsquemasPorFuente:
    def test_uci_acepta_consumo_plausible(self) -> None:
        uci_electricity_schema().validate(_frame(y=[100.0, 200.0, 300.0]), lazy=True)

    def test_ree_rechaza_demanda_negativa(self) -> None:
        with pytest.raises(pandera.errors.SchemaErrors):
            ree_demand_schema().validate(_frame(y=[-10.0, 200.0, 300.0]), lazy=True)

    def test_open_meteo_acepta_temperaturas_extremas_plausibles(self) -> None:
        frame = _frame()
        frame = frame.rename(columns={"y": "temp_c"})
        frame["temp_c"] = [-30.0, 45.0, 10.0]
        open_meteo_schema().validate(frame, lazy=True)

    def test_open_meteo_rechaza_temperatura_imposible(self) -> None:
        frame = _frame().rename(columns={"y": "temp_c"})
        frame["temp_c"] = [-30.0, 200.0, 10.0]  # error de unidad tipico (Fahrenheit)
        with pytest.raises(pandera.errors.SchemaErrors):
            open_meteo_schema().validate(frame, lazy=True)

    def test_binance_acepta_precio_de_cierre(self) -> None:
        binance_schema().validate(_frame(y=[42_000.0, 42_100.0, 41_900.0]), lazy=True)
