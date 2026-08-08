"""Lectura desde disco: `ARTIFACT_FILES`, `available_artifacts` y los `load_*`.

`scoring_frame` (la variante en memoria) tiene su propia suite en
`test_reader.py`; este fichero cubre exclusivamente la variante que consume
`chronolab.app`: leer, validar contra `chronolab.artifacts.schemas` y fallar
con un mensaje util cuando el fichero no existe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chronolab.artifacts import reader
from chronolab.errors import ArtifactNotFound


def _panel_frame() -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=48, freq="h")
    return pd.DataFrame(
        {
            "unique_id": ["s00"] * 48,
            "ds": index,
            "y": range(48),
            "temp_c": [12.0] * 48,
            "voltage": [230.0] * 48,
        }
    )


class TestAvailableArtifacts:
    def test_ninguno_disponible_en_un_directorio_vacio(self, tmp_path: Path) -> None:
        status = reader.available_artifacts(tmp_path)
        assert set(status) == set(reader.ARTIFACT_FILES)
        assert not any(status.values())

    def test_detecta_el_fichero_que_si_existe(self, tmp_path: Path) -> None:
        _panel_frame().to_parquet(tmp_path / reader.ARTIFACT_FILES["panel"], index=False)
        status = reader.available_artifacts(tmp_path)
        assert status["panel"] is True
        assert status["leaderboard"] is False


class TestLoadPanel:
    def test_lee_y_valida_un_panel_bien_formado(self, tmp_path: Path) -> None:
        frame = _panel_frame()
        frame.to_parquet(tmp_path / "panel.parquet", index=False)
        loaded = reader.load_panel(results_dir=tmp_path)
        assert list(loaded.columns) == ["unique_id", "ds", "y", "temp_c", "voltage"]
        assert len(loaded) == 48

    def test_fichero_ausente_lanza_artifact_not_found_con_mensaje_util(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ArtifactNotFound, match="panel"):
            reader.load_panel(results_dir=tmp_path)

    def test_columna_extra_en_un_esquema_cerrado_se_rechaza(self, tmp_path: Path) -> None:
        # panel_schema() es estricto (lo genera solo build_demo_artifacts.py): una
        # columna que nadie declaro no debe colarse en silencio.
        frame = _panel_frame()
        frame["columna_inesperada"] = 1.0
        frame.to_parquet(tmp_path / "panel.parquet", index=False)
        with pytest.raises(Exception, match=r"columna_inesperada|unexpected|not in schema"):
            reader.load_panel(results_dir=tmp_path)


class TestLoadLeaderboard:
    def test_columna_extra_en_un_esquema_abierto_no_revienta(self, tmp_path: Path) -> None:
        # leaderboard_schema() es strict=False: lo produce un script externo y no
        # debe romperse si ese script anade una columna nueva.
        frame = pd.DataFrame(
            {
                "model_id": ["naive"],
                "unique_id": [None],
                "stage": ["holdout"],
                "n_windows": [2],
                "mae": [1.0],
                "rmse": [1.0],
                "mape": [1.0],
                "smape": [1.0],
                "mase": [1.0],
                "coverage_95": [0.9],
                "coverage_80": [0.8],
                "coverage_50": [0.5],
                "fit_seconds_total": [0.1],
                "predict_seconds_total": [0.1],
                "is_zero_shot": [False],
                "training_regime": ["fitted"],  # columna que el esquema no declara
            }
        )
        frame.to_parquet(tmp_path / "leaderboard.parquet", index=False)
        loaded = reader.load_leaderboard(results_dir=tmp_path)
        assert "training_regime" in loaded.columns


class TestLoadDmMatrix:
    def test_p_value_fuera_de_rango_se_rechaza(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "model_a": ["naive"],
                "model_b": ["mstl"],
                "stat": [1.2],
                "p_value": [1.5],  # invalido: fuera de [0, 1]
                "n_obs": [100],
                "mean_difference": [0.1],
                "degenerate": [False],
            }
        )
        frame.to_parquet(tmp_path / "dm_matrix.parquet", index=False)
        with pytest.raises(Exception, match=r"in_range|1\.5"):
            reader.load_dm_matrix(results_dir=tmp_path)

    def test_par_valido_se_lee_tal_cual(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "model_a": ["naive"],
                "model_b": ["mstl"],
                "stat": [1.2],
                "p_value": [0.05],
                "n_obs": [100],
                "mean_difference": [0.1],
                "degenerate": [False],
            }
        )
        frame.to_parquet(tmp_path / "dm_matrix.parquet", index=False)
        loaded = reader.load_dm_matrix(results_dir=tmp_path)
        assert loaded.loc[0, "model_a"] == "naive"


class TestArtifactFilesUniveral:
    def test_cada_artefacto_de_artifact_files_tiene_un_load(self) -> None:
        # Barrera contra el desfase entre `ARTIFACT_FILES` y los `load_*` -si
        # alguien anade un artefacto nuevo aqui y olvida el loader, este test
        # lo dice en vez de dejar que la app falle en produccion con un
        # `AttributeError` al llamar `state.load_<lo que sea>`.
        for name in reader.ARTIFACT_FILES:
            assert hasattr(reader, f"load_{name}"), f"falta reader.load_{name}"
