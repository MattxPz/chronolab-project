"""Inyeccion de anomalias sinteticas: los seis tipos, su verdad y su reproducibilidad."""

from __future__ import annotations

import pandas as pd
import pytest

from chronolab.anomaly.injection import KINDS, TRUTH_COLUMNS, AnomalySpec, inject_anomalies
from chronolab.panel import Panel
from tests.fixtures.synthetic import make_hourly_panel


@pytest.fixture(scope="module")
def panel() -> Panel:
    return make_hourly_panel(n_series=2, n_hours=24 * 30)


@pytest.fixture(scope="module")
def uid(panel: Panel) -> str:
    return str(panel.ids()[0])


@pytest.fixture(scope="module")
def start(panel: Panel) -> pd.Timestamp:
    return panel.first_ds + pd.Timedelta(hours=200)


class TestAnomalySpec:
    def test_kind_invalido_falla(self, uid: str, start: pd.Timestamp) -> None:
        with pytest.raises(ValueError):
            AnomalySpec(kind="not_a_kind", unique_id=uid, start=start, duration=1, magnitude=1.0)  # type: ignore[arg-type]

    def test_duracion_menor_que_uno_falla(self, uid: str, start: pd.Timestamp) -> None:
        with pytest.raises(ValueError):
            AnomalySpec(kind="spike", unique_id=uid, start=start, duration=0, magnitude=1.0)

    @pytest.mark.parametrize("magnitude", [0.0, -0.1, 1.5])
    def test_data_gap_exige_magnitude_en_0_1(
        self, uid: str, start: pd.Timestamp, magnitude: float
    ) -> None:
        with pytest.raises(ValueError):
            AnomalySpec(
                kind="data_gap", unique_id=uid, start=start, duration=3, magnitude=magnitude
            )

    def test_variance_shift_exige_magnitude_positiva(self, uid: str, start: pd.Timestamp) -> None:
        with pytest.raises(ValueError):
            AnomalySpec(
                kind="variance_shift", unique_id=uid, start=start, duration=3, magnitude=0.0
            )

    def test_seasonal_phase_exige_al_menos_un_paso(self, uid: str, start: pd.Timestamp) -> None:
        with pytest.raises(ValueError):
            AnomalySpec(
                kind="seasonal_phase", unique_id=uid, start=start, duration=3, magnitude=0.5
            )


class TestInyeccion:
    def test_los_seis_tipos_producen_verdad(
        self, panel: Panel, uid: str, start: pd.Timestamp
    ) -> None:
        specs = [
            AnomalySpec(kind="spike", unique_id=uid, start=start, duration=2, magnitude=6.0),
            AnomalySpec(
                kind="level_shift",
                unique_id=uid,
                start=start + pd.Timedelta(hours=20),
                duration=10,
                magnitude=4.0,
            ),
            AnomalySpec(
                kind="variance_shift",
                unique_id=uid,
                start=start + pd.Timedelta(hours=50),
                duration=15,
                magnitude=5.0,
            ),
            AnomalySpec(
                kind="seasonal_phase",
                unique_id=uid,
                start=start + pd.Timedelta(hours=100),
                duration=12,
                magnitude=24.0,
            ),
            AnomalySpec(
                kind="sensor_freeze",
                unique_id=uid,
                start=start + pd.Timedelta(hours=150),
                duration=8,
                magnitude=0.0,
            ),
            AnomalySpec(
                kind="data_gap",
                unique_id=uid,
                start=start + pd.Timedelta(hours=180),
                duration=6,
                magnitude=1.0,
            ),
        ]
        contaminated, truth = inject_anomalies(panel, specs, seed=0)

        assert list(truth.columns) == list(TRUTH_COLUMNS)
        assert set(truth["anomaly_type"]) == set(KINDS)
        assert bool(truth["is_anomaly"].all())
        assert len(contaminated.df) == len(panel.df)

    def test_data_gap_deja_nan_reales(self, panel: Panel, uid: str, start: pd.Timestamp) -> None:
        spec = AnomalySpec(kind="data_gap", unique_id=uid, start=start, duration=5, magnitude=1.0)
        contaminated, truth = inject_anomalies(panel, [spec], seed=1)

        span = pd.date_range(start, periods=5, freq=panel.spec.freq)
        values = (
            contaminated.df.loc[contaminated.df["unique_id"] == uid].set_index("ds")["y"].loc[span]
        )
        assert values.isna().all()
        assert truth["severity"].isna().all()

    def test_sensor_freeze_deja_un_valor_constante(
        self, panel: Panel, uid: str, start: pd.Timestamp
    ) -> None:
        spec = AnomalySpec(
            kind="sensor_freeze", unique_id=uid, start=start, duration=6, magnitude=0.0
        )
        contaminated, _ = inject_anomalies(panel, [spec], seed=2)

        span = pd.date_range(start, periods=6, freq=panel.spec.freq)
        values = (
            contaminated.df.loc[contaminated.df["unique_id"] == uid].set_index("ds")["y"].loc[span]
        )
        assert values.nunique() == 1

    def test_misma_seed_produce_el_mismo_resultado(
        self, panel: Panel, uid: str, start: pd.Timestamp
    ) -> None:
        specs = [AnomalySpec(kind="spike", unique_id=uid, start=start, duration=3, magnitude=6.0)]
        c1, t1 = inject_anomalies(panel, specs, seed=7)
        c2, t2 = inject_anomalies(panel, specs, seed=7)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_series_equal(c1.df["y"].astype(float), c2.df["y"].astype(float))

    def test_el_panel_original_no_se_modifica(
        self, panel: Panel, uid: str, start: pd.Timestamp
    ) -> None:
        before = panel.df["y"].copy()
        specs = [AnomalySpec(kind="spike", unique_id=uid, start=start, duration=3, magnitude=6.0)]
        inject_anomalies(panel, specs, seed=3)
        pd.testing.assert_series_equal(before, panel.df["y"])

    def test_serie_inexistente_falla(self, panel: Panel, start: pd.Timestamp) -> None:
        specs = [
            AnomalySpec(kind="spike", unique_id="no_existe", start=start, duration=3, magnitude=6.0)
        ]
        with pytest.raises(ValueError, match="no_existe"):
            inject_anomalies(panel, specs, seed=0)

    def test_tramo_fuera_de_rango_falla(self, panel: Panel, uid: str) -> None:
        specs = [
            AnomalySpec(kind="spike", unique_id=uid, start=panel.last_ds, duration=5, magnitude=6.0)
        ]
        with pytest.raises(ValueError, match="rango observado"):
            inject_anomalies(panel, specs, seed=0)
