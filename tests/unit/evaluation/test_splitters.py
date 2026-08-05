"""`Window` y `RollingOriginSplitter`: invariantes y validacion de configuracion."""

from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from chronolab.errors import ShortTrainWarning, WindowValidationError
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


class TestSplit:
    def test_genera_las_ventanas_pedidas_ancladas_al_final_del_panel(
        self, hourly_panel: Panel
    ) -> None:
        windows = RollingOriginSplitter(h=24, n_windows=5, step_size=24).split(hourly_panel)

        assert len(windows) == 5
        assert [w.window_id for w in windows] == [0, 1, 2, 3, 4]
        # El ancla es el final: la ultima ventana evalua hasta el ultimo dato.
        assert windows[-1].last_pred == hourly_panel.last_ds

    def test_los_cutoffs_se_separan_exactamente_step_size(self, hourly_panel: Panel) -> None:
        windows = RollingOriginSplitter(h=24, n_windows=4, step_size=48).split(hourly_panel)
        separations = {b.cutoff - a.cutoff for a, b in pairwise(windows)}
        assert separations == {pd.Timedelta(hours=48)}

    def test_el_modo_expansivo_mantiene_el_inicio_del_entrenamiento(
        self, hourly_panel: Panel
    ) -> None:
        windows = RollingOriginSplitter(h=24, n_windows=4, step_size=24).split(hourly_panel)
        assert {w.train_start for w in windows} == {hourly_panel.first_ds}

    def test_el_modo_deslizante_mantiene_la_longitud_del_entrenamiento(
        self, hourly_panel: Panel
    ) -> None:
        splitter = RollingOriginSplitter(
            h=24, n_windows=4, step_size=24, mode="sliding", train_size=336
        )
        windows = splitter.split(hourly_panel)

        lengths = {w.cutoff - w.train_start for w in windows}
        assert lengths == {pd.Timedelta(hours=335)}  # 336 pasos inclusivos
        # Y el inicio se mueve, que es lo que lo distingue del expansivo.
        assert len({w.train_start for w in windows}) == 4

    def test_el_gap_separa_el_entrenamiento_de_la_evaluacion(self, hourly_panel: Panel) -> None:
        windows = RollingOriginSplitter(h=24, n_windows=3, step_size=24, gap=6).split(hourly_panel)
        for window in windows:
            assert window.first_pred - window.cutoff == pd.Timedelta(hours=7)
            assert window.last_pred - window.first_pred == pd.Timedelta(hours=23)

    def test_las_ultimas_ventanas_son_de_holdout(self, hourly_panel: Panel) -> None:
        windows = RollingOriginSplitter(h=24, n_windows=5, step_size=24, holdout_windows=2).split(
            hourly_panel
        )
        assert [w.stage for w in windows] == ["dev", "dev", "dev", "holdout", "holdout"]

    def test_descarta_con_aviso_las_ventanas_sin_entrenamiento_suficiente(
        self, hourly_panel: Panel
    ) -> None:
        # El panel tiene 2016 pasos; con h=24 y step_size=168 el cutoff mas
        # antiguo cae en el paso 984, asi que exigir 1500 de entrenamiento deja
        # fuera las primeras ventanas en lugar de recortarlas.
        splitter = RollingOriginSplitter(h=24, n_windows=6, step_size=168, min_context=1500)
        with pytest.warns(ShortTrainWarning, match="entrenamiento insuficiente"):
            windows = splitter.split(hourly_panel)

        assert 0 < len(windows) < 6
        assert all(w.cutoff - w.train_start >= pd.Timedelta(hours=1499) for w in windows)
        # Renumeradas desde cero y contiguas: `windows.parquet` no tiene huecos.
        assert [w.window_id for w in windows] == list(range(len(windows)))

    def test_el_holdout_no_se_desplaza_al_descartar_ventanas(self, hourly_panel: Panel) -> None:
        # El holdout se decide sobre la numeracion del plan: descartar ventanas
        # antiguas no puede cambiar cuales son las que se reportan.
        splitter = RollingOriginSplitter(
            h=24, n_windows=6, step_size=168, min_context=1500, holdout_windows=2
        )
        with pytest.warns(ShortTrainWarning):
            windows = splitter.split(hourly_panel)

        assert [w.stage for w in windows[-2:]] == ["holdout", "holdout"]
        assert all(w.stage == "dev" for w in windows[:-2])

    def test_rechaza_un_panel_que_no_da_para_una_ventana(self, hourly_panel: Panel) -> None:
        short = hourly_panel.slice(
            hourly_panel.first_ds, hourly_panel.first_ds + pd.Timedelta(hours=10)
        )
        with pytest.raises(WindowValidationError, match="el panel tiene 11 pasos"):
            RollingOriginSplitter(h=24, n_windows=1).split(short)

    def test_rechaza_un_plan_cuyas_ventanas_no_caben_todas_por_historia(
        self, hourly_panel: Panel
    ) -> None:
        splitter = RollingOriginSplitter(
            h=24, n_windows=2, mode="sliding", train_size=5000, min_context=5000
        )
        with pytest.raises(WindowValidationError, match="ninguna de las 2 ventanas"):
            splitter.split(hourly_panel)

    def test_el_entrenamiento_nunca_alcanza_al_tramo_evaluado(self, hourly_panel: Panel) -> None:
        # La propiedad central del origen rodante, comprobada aqui sobre el caso
        # mas peligroso: step_size < h, es decir tramos de evaluacion solapados
        # entre ventanas. El solape entre ventanas es legitimo; el solape entre
        # el train y el test de la *misma* ventana nunca lo es.
        windows = RollingOriginSplitter(h=48, n_windows=6, step_size=12).split(hourly_panel)
        for window in windows:
            assert window.train_start <= window.cutoff < window.first_pred <= window.last_pred
