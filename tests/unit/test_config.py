from __future__ import annotations

from pathlib import Path

import pytest

from chronolab.config import AppSettings, figures_dir, get_settings, results_dir


class TestAppSettings:
    def test_valores_por_defecto_apuntan_dentro_del_repo(self) -> None:
        settings = AppSettings()
        assert settings.results_dir.name == "results"
        assert settings.results_dir.parent.name == "reports"
        assert settings.figures_dir.name == "figures"
        assert settings.demo_mode is True

    def test_variable_de_entorno_sobrescribe_la_ruta(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CHRONOLAB_RESULTS_DIR", str(tmp_path))
        settings = AppSettings()
        assert settings.results_dir == tmp_path

    def test_es_inmutable(self) -> None:
        settings = AppSettings()
        with pytest.raises(Exception, match=r"frozen|immutable"):
            settings.demo_mode = False


class TestHelpers:
    def test_results_dir_y_figures_dir_usan_get_settings(self) -> None:
        assert results_dir() == get_settings().results_dir
        assert figures_dir() == get_settings().figures_dir

    def test_get_settings_esta_cacheado(self) -> None:
        assert get_settings() is get_settings()
