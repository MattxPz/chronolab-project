"""`Window` y `RollingOriginSplitter`: invariantes y validacion de configuracion."""

from __future__ import annotations

import pandas as pd
import pytest

from chronolab.errors import WindowValidationError
from chronolab.evaluation.splitters import RollingOriginSplitter, Window
from chronolab.panel import Panel

BASE = pd.Timestamp("2023-01-02")


def _window(**overrides: object) -> Window:
    cutoff = BASE + pd.Timedelta(hours=999)
    kwargs: dict[str, object] = {
        "window_id": 0,
        "stage": "dev",
        "train_start": BASE,
        "cutoff": cutoff,
        "first_pred": cutoff + pd.Timedelta(hours=1),
        "last_pred": cutoff + pd.Timedelta(hours=24),
        "h": 24,
        "gap": 0,
    }
    kwargs.update(overrides)
    return Window(**kwargs)  # type: ignore[arg-type]


class TestWindow:
    def test_acepta_una_ventana_coherente(self) -> None:
        window = _window()
        assert window.cutoff < window.first_pred <= window.last_pred

    def test_rechaza_que_la_prediccion_empiece_en_el_cutoff(self) -> None:
        # El off-by-one clasico: si `first_pred == cutoff`, el primer instante
        # evaluado ya era conocido.
        cutoff = BASE + pd.Timedelta(hours=999)
        with pytest.raises(WindowValidationError, match="no es posterior a cutoff"):
            _window(first_pred=cutoff)

    def test_rechaza_que_la_prediccion_caiga_antes_del_cutoff(self) -> None:
        cutoff = BASE + pd.Timedelta(hours=999)
        with pytest.raises(WindowValidationError, match="no es posterior a cutoff"):
            _window(first_pred=cutoff - pd.Timedelta(hours=5))

    def test_rechaza_train_start_posterior_al_cutoff(self) -> None:
        with pytest.raises(WindowValidationError, match="posterior a cutoff"):
            _window(train_start=BASE + pd.Timedelta(hours=5000))

    def test_rechaza_last_pred_anterior_a_first_pred(self) -> None:
        cutoff = BASE + pd.Timedelta(hours=999)
        with pytest.raises(WindowValidationError, match="anterior a first_pred"):
            _window(last_pred=cutoff + pd.Timedelta(minutes=30))

    @pytest.mark.parametrize(("field", "value"), [("h", 0), ("gap", -1), ("window_id", -1)])
    def test_rechaza_parametros_negativos(self, field: str, value: int) -> None:
        with pytest.raises(WindowValidationError):
            _window(**{field: value})

    def test_es_inmutable(self) -> None:
        with pytest.raises(AttributeError):
            _window().h = 48  # type: ignore[misc]


class TestLead:
    def test_sin_gap_el_adelanto_coincide_con_el_paso(self) -> None:
        window = _window()
        assert window.lead(1) == 1
        assert window.lead(24) == 24

    def test_con_gap_el_adelanto_se_desplaza(self) -> None:
        # `h_step` es relativo a `first_pred`; el adelanto real desde el cutoff
        # incluye el gap, y esa distincion es la que decide que features tienen
        # `max_lead` suficiente.
        cutoff = BASE + pd.Timedelta(hours=999)
        window = _window(
            gap=6,
            first_pred=cutoff + pd.Timedelta(hours=7),
            last_pred=cutoff + pd.Timedelta(hours=30),
        )
        assert window.lead(1) == 7
        assert window.lead(24) == 30

    @pytest.mark.parametrize("h_step", [0, -1, 25])
    def test_rechaza_pasos_fuera_del_horizonte(self, h_step: int) -> None:
        with pytest.raises(ValueError, match="h_step fuera de"):
            _window().lead(h_step)


class TestRollingOriginSplitter:
    def test_acepta_una_configuracion_coherente(self) -> None:
        splitter = RollingOriginSplitter(h=24, n_windows=10, step_size=24, gap=0)
        assert splitter.mode == "expanding"
        assert splitter.holdout_windows == 0

    def test_el_modo_deslizante_exige_train_size(self) -> None:
        with pytest.raises(WindowValidationError, match="exige train_size"):
            RollingOriginSplitter(h=24, n_windows=5, mode="sliding")

    def test_el_modo_deslizante_con_train_size_es_valido(self) -> None:
        splitter = RollingOriginSplitter(h=24, n_windows=5, mode="sliding", train_size=2000)
        assert splitter.train_size == 2000

    @pytest.mark.parametrize(
        ("field", "value"),
        [("h", 0), ("n_windows", 0), ("step_size", 0), ("gap", -1), ("min_context", 0)],
    )
    def test_rechaza_parametros_invalidos(self, field: str, value: int) -> None:
        kwargs: dict[str, object] = {"h": 24, "n_windows": 5}
        kwargs[field] = value
        with pytest.raises(WindowValidationError):
            RollingOriginSplitter(**kwargs)  # type: ignore[arg-type]

    def test_rechaza_mas_holdout_que_ventanas(self) -> None:
        with pytest.raises(WindowValidationError, match="holdout_windows fuera de"):
            RollingOriginSplitter(h=24, n_windows=5, holdout_windows=6)

    def test_rechaza_train_size_menor_que_min_context(self) -> None:
        with pytest.raises(WindowValidationError, match="menor que min_context"):
            RollingOriginSplitter(
                h=24, n_windows=5, mode="sliding", train_size=100, min_context=200
            )

    def test_no_admite_particionar_por_mascara(self) -> None:
        # Es el unico emisor de particiones del proyecto y no expone ninguna via
        # que acepte indices arbitrarios: sin esa via, un split aleatorio sobre
        # datos temporales no se puede escribir por accidente.
        splitter = RollingOriginSplitter(h=24, n_windows=5)
        for forbidden in ("split_mask", "from_indices", "train_test_split", "sample"):
            assert not hasattr(splitter, forbidden)

    def test_split_esta_pendiente_de_implementar(self, hourly_panel: Panel) -> None:
        splitter = RollingOriginSplitter(h=24, n_windows=5)
        with pytest.raises(NotImplementedError):
            splitter.split(hourly_panel)
