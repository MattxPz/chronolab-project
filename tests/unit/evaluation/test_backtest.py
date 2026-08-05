"""Motor de backtesting: esquema del artefacto, politica de refit y tratamiento de fallos.

Los tests de fuga viven aparte, en `tests/leakage/`. Aqui se comprueba que el
motor produce el artefacto que dice producir y que un modelo roto se ve.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from chronolab.data.futr import RealizedFutrProvider
from chronolab.errors import (
    ChronolabError,
    CutoffViolation,
    LeakageError,
    MissingFutrExog,
    PerfectForesightWarning,
)
from chronolab.evaluation.backtest import (
    FORECAST_KEY_COLUMNS,
    BacktestPlan,
    BacktestResult,
    backtest,
)
from chronolab.evaluation.splitters import Window
from chronolab.models.protocols import QUANTILES, ModelRequirements, quantile_column
from chronolab.panel import FutrFrame, Panel, PanelSpec
from chronolab.types import DatasetId, ModelId, Vintage
from tests.fixtures.models import (
    CrossedQuantileProbe,
    CutoffViolatingProbe,
    FailingProbe,
    ScaledExogProbe,
    SeasonalNaiveProbe,
)

PLAN = BacktestPlan(h=24, n_windows=3, step_size=24)
MALFORMED_ID = ModelId("probe_malformado")
LYING_CUTOFF_ID = ModelId("probe_cutoff_mentiroso")
PLAIN_REQUIREMENTS = ModelRequirements()


@pytest.fixture
def realized_futr(hourly_panel: Panel) -> RealizedFutrProvider:
    with pytest.warns(PerfectForesightWarning):
        return RealizedFutrProvider(panel=hourly_panel)


def _run(panel: Panel, models: list[object], **kwargs: object) -> BacktestResult:
    return backtest(panel, models, PLAN, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _ManipulatedProvider:
    """Proveedor que devuelve una trama de futuras manipulada.

    El motor no da por bueno lo que le llega de un colaborador: un proveedor mal
    escrito, o escrito por otra persona, es exactamente el escenario en el que
    una comprobacion sobra hasta el dia en que no sobra.
    """

    inner: RealizedFutrProvider
    kind: str

    @property
    def vintage(self) -> Vintage:
        return self.inner.vintage

    def futr(self, window: Window, ids: Sequence[str]) -> FutrFrame:
        futr = self.inner.futr(window, ids=ids)
        frame = futr.df.copy()

        if self.kind == "target":
            frame["y"] = 1.0
        elif self.kind == "hist":
            frame["voltage"] = 230.0
        elif self.kind == "missing":
            frame = frame.drop(columns=["temp_c"])
        elif self.kind == "past":
            frame["ds"] = frame["ds"] - pd.Timedelta(hours=1_000)
        elif self.kind == "otra_ventana":
            otra = Window(
                window_id=window.window_id + 99,
                stage=window.stage,
                train_start=window.train_start,
                cutoff=window.cutoff,
                first_pred=window.first_pred,
                last_pred=window.last_pred,
                h=window.h,
                gap=window.gap,
            )
            return FutrFrame(df=frame, window=otra, vintage=futr.vintage)

        return FutrFrame(df=frame, window=window, vintage=futr.vintage)


@dataclass(frozen=True, slots=True)
class _FittedMalformed:
    """Ajuste que devuelve predicciones que incumplen el contrato de `predict`."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    ids: tuple[str, ...]
    kind: str

    @property
    def n_params(self) -> int | None:
        return None

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:]
        frame = pd.DataFrame(
            {
                "unique_id": np.repeat(list(self.ids), self.h),
                "ds": np.tile(grid.to_numpy(), len(self.ids)),
                "y_hat": 0.0,
            }
        )

        if self.kind == "sin_y_hat":
            return frame.drop(columns=["y_hat"])
        if self.kind == "ds_no_temporal":
            return frame.assign(ds=frame["ds"].astype(str))
        if self.kind == "ds_con_huso":
            # UTC y no un huso civil: localizar en Europe/Madrid reventaria en la
            # hora que no existe del cambio de marzo, y el test dejaria de medir
            # lo que dice medir.
            return frame.assign(ds=frame["ds"].dt.tz_localize("UTC"))
        if self.kind == "faltan_filas":
            return frame.iloc[:-1]
        if self.kind == "serie_de_mas":
            extra = frame.head(self.h).assign(unique_id="serie_inventada")
            return pd.concat([frame, extra], ignore_index=True)
        if self.kind == "duplicadas":
            return pd.concat([frame.iloc[:-1], frame.iloc[:1]], ignore_index=True)
        if self.kind == "fuera_de_rejilla":
            return frame.assign(ds=frame["ds"] + pd.Timedelta(minutes=30))
        raise AssertionError(f"tipo de deformidad desconocido: {self.kind}")


@dataclass(frozen=True)
class _MalformedProbe:
    """Modelo cuya prediccion incumple el contrato de una manera concreta."""

    kind: str
    model_id: ModelId = MALFORMED_ID
    requires: ModelRequirements = PLAIN_REQUIREMENTS

    def fit(self, train: Panel, *, h: int) -> _FittedMalformed:
        return _FittedMalformed(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=0.0,
            freq=train.spec.freq,
            ids=tuple(str(uid) for uid in train.ids()),
            kind=self.kind,
        )


@dataclass(frozen=True, slots=True)
class _FittedLyingCutoff:
    """Ajuste que declara un cutoff posterior al de su ventana."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float

    @property
    def n_params(self) -> int | None:
        return None

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:  # pragma: no cover  el motor lo detiene antes
        raise AssertionError("no deberia llegar a predecir")


@dataclass(frozen=True)
class _LyingCutoffProbe:
    """Modelo que dice estar ajustado mas alla del cutoff de su ventana."""

    model_id: ModelId = LYING_CUTOFF_ID
    requires: ModelRequirements = PLAIN_REQUIREMENTS

    def fit(self, train: Panel, *, h: int) -> _FittedLyingCutoff:
        return _FittedLyingCutoff(
            model_id=self.model_id,
            cutoff=train.last_ds + pd.Timedelta(hours=1_000),
            h=h,
            fit_seconds=0.0,
        )


class TestEsquemaDelArtefacto:
    def test_forecasts_tiene_el_esquema_declarado(self, hourly_panel: Panel) -> None:
        result = backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN)

        assert list(result.forecasts.columns) == [
            *FORECAST_KEY_COLUMNS,
            *[quantile_column(q) for q in QUANTILES],
        ]
        assert result.forecasts["window_id"].dtype == np.int16
        assert result.forecasts["h_step"].dtype == np.int16
        for column in ("y", "y_hat", "q_5000"):
            assert result.forecasts[column].dtype == np.float32
        assert result.forecasts["ds"].dtype == np.dtype("datetime64[ns]")
        assert result.forecasts["cutoff"].dtype == np.dtype("datetime64[ns]")

    def test_una_fila_por_serie_ventana_e_instante(self, hourly_panel: Panel) -> None:
        result = backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN)

        assert len(result.forecasts) == 3 * len(hourly_panel.ids()) * PLAN.h
        assert not result.forecasts.duplicated(
            subset=["model_id", "unique_id", "window_id", "ds"]
        ).any()

    def test_h_step_es_relativo_a_la_primera_prediccion(self, hourly_panel: Panel) -> None:
        plan = BacktestPlan(h=6, n_windows=2, step_size=24, gap=3)
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=hourly_panel)
        result = backtest(hourly_panel, [SeasonalNaiveProbe()], plan, futr=provider)

        assert sorted(result.forecasts["h_step"].unique()) == [1, 2, 3, 4, 5, 6]
        first = result.forecasts[result.forecasts["h_step"] == 1]
        cutoffs = result.forecasts.groupby("window_id")["cutoff"].first()
        for window_id, group in first.groupby("window_id"):
            # El adelanto real del primer paso es gap + 1, no 1.
            assert (group["ds"] - cutoffs[window_id] == pd.Timedelta(hours=4)).all()

    def test_y_viaja_desnormalizada_y_coincide_con_el_panel(self, hourly_panel: Panel) -> None:
        result = backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN)
        observed = hourly_panel.df[["unique_id", "ds", "y"]]

        merged = result.forecasts.merge(
            observed, on=["unique_id", "ds"], suffixes=("_artifact", "_panel")
        )
        assert len(merged) == len(result.forecasts)
        np.testing.assert_allclose(merged["y_artifact"], merged["y_panel"], rtol=1e-6)

    def test_los_cuantiles_no_soportados_son_nan_y_no_un_intervalo_inventado(
        self, hourly_panel: Panel
    ) -> None:
        result = backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN)
        for quantile in QUANTILES:
            assert result.forecasts[quantile_column(quantile)].isna().all()

    def test_windows_describe_cada_ventana_efectiva(self, hourly_panel: Panel) -> None:
        plan = BacktestPlan(h=24, n_windows=3, step_size=24, holdout_windows=1)
        result = backtest(hourly_panel, [SeasonalNaiveProbe()], plan)

        assert list(result.windows.columns) == [
            "window_id",
            "stage",
            "train_start",
            "cutoff",
            "first_pred",
            "last_pred",
            "n_train_obs",
            "n_series",
        ]
        assert result.windows["stage"].tolist() == ["dev", "dev", "holdout"]
        assert (result.windows["n_series"] == len(hourly_panel.ids())).all()
        # Entrenamiento expansivo: cada ventana ve mas observaciones que la anterior.
        assert result.windows["n_train_obs"].is_monotonic_increasing

    def test_el_resultado_lleva_el_vintage_de_las_exogenas(
        self, hourly_panel: Panel, realized_futr: RealizedFutrProvider
    ) -> None:
        # Comparar filas de vintages distintos no es legitimo, asi que el vintage
        # viaja pegado al resultado en lugar de reconstruirse despues.
        assert backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN).futr_vintage is None
        with_futr = backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN, futr=realized_futr)
        assert with_futr.futr_vintage == "realized"


class TestPrediccionesCorrectas:
    def test_el_naive_estacional_reproduce_el_valor_de_hace_una_estacion(
        self, hourly_panel: Panel
    ) -> None:
        plan = BacktestPlan(h=24, n_windows=1)
        result = backtest(hourly_panel, [SeasonalNaiveProbe(season=24)], plan)

        window_cutoff = result.windows.loc[0, "cutoff"]
        expected = hourly_panel.df[
            (hourly_panel.df["ds"] > window_cutoff - pd.Timedelta(hours=24))
            & (hourly_panel.df["ds"] <= window_cutoff)
        ]
        merged = result.forecasts.merge(
            expected.assign(ds=expected["ds"] + pd.Timedelta(hours=24))[
                ["unique_id", "ds", "y"]
            ].rename(columns={"y": "y_esperada"}),
            on=["unique_id", "ds"],
        )
        assert len(merged) == len(result.forecasts)
        np.testing.assert_allclose(merged["y_hat"], merged["y_esperada"], rtol=1e-5)


class TestPoliticaDeRefit:
    def test_los_modelos_baratos_se_reajustan_en_cada_ventana(self, hourly_panel: Panel) -> None:
        result = backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN)
        assert result.model_runs["refit"].tolist() == [True, True, True]
        assert result.model_runs["refit_every"].tolist() == [1, 1, 1]

    def test_los_modelos_caros_se_ajustan_una_sola_vez(
        self, hourly_panel: Panel, realized_futr: RealizedFutrProvider
    ) -> None:
        expensive = SeasonalNaiveProbe(
            model_id=ModelId("probe_caro"),
            requires=ModelRequirements(min_context=24, refit_cost="expensive"),
        )
        result = backtest(hourly_panel, [expensive], PLAN, futr=realized_futr)

        assert result.model_runs["refit"].tolist() == [True, False, False]
        assert result.model_runs["refit_every"].tolist() == [3, 3, 3]
        # Reutilizar es obsolescencia, no fuga: el ajuste es anterior a la ventana.
        assert (result.model_runs.loc[1:, "fit_seconds"] == 0).all()

    def test_la_politica_explicita_manda_sobre_la_del_coste(
        self, hourly_panel: Panel, realized_futr: RealizedFutrProvider
    ) -> None:
        plan = BacktestPlan(h=24, n_windows=4, step_size=24, refit_every=2)
        result = backtest(hourly_panel, [SeasonalNaiveProbe()], plan, futr=realized_futr)
        assert result.model_runs["refit"].tolist() == [True, False, True, False]

    def test_un_ajuste_reutilizado_predice_desde_su_propio_cutoff(
        self, hourly_panel: Panel, realized_futr: RealizedFutrProvider
    ) -> None:
        # La reutilizacion es legitima —el ajuste solo conoce datos anteriores a
        # su cutoff, que es anterior al de la ventana— pero cambia el resultado.
        # Por eso queda registrada en lugar de aplicarse en silencio.
        expensive = SeasonalNaiveProbe(
            model_id=ModelId("probe_caro"),
            requires=ModelRequirements(min_context=24, refit_cost="expensive"),
        )
        stale = backtest(hourly_panel, [expensive], PLAN, futr=realized_futr)
        fresh = backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN, futr=realized_futr)

        last_window = stale.forecasts["window_id"] == 2
        assert not np.allclose(
            stale.forecasts.loc[last_window, "y_hat"].to_numpy(),
            fresh.forecasts.loc[last_window, "y_hat"].to_numpy(),
        )


class TestFallosVisibles:
    def test_un_modelo_que_revienta_ocupa_una_fila_y_no_desaparece(
        self, hourly_panel: Panel
    ) -> None:
        result = backtest(hourly_panel, [SeasonalNaiveProbe(), FailingProbe()], PLAN)

        failed = result.model_runs[result.model_runs["model_id"] == "probe_failing"]
        assert failed["status"].tolist() == ["failed"] * 3
        assert failed["error"].str.startswith("RuntimeError").all()
        # No ha escrito predicciones, pero el leaderboard sabra que existio.
        assert "probe_failing" not in set(result.forecasts["model_id"])
        assert set(result.forecasts["model_id"]) == {"probe_seasonal_naive"}

    def test_un_modelo_falla_solo_en_las_ventanas_en_que_falla(self, hourly_panel: Panel) -> None:
        windows = PLAN.splitter().split(hourly_panel)
        model = FailingProbe(fail_on=(windows[1].cutoff,))
        result = backtest(hourly_panel, [model], PLAN)

        assert result.model_runs["status"].tolist() == ["ok", "failed", "ok"]
        # Sus metricas se calcularan sobre dos ventanas, y `n_obs` lo delatara.
        assert set(result.forecasts["window_id"]) == {0, 2}

    def test_la_ventana_sin_contexto_suficiente_se_salta_con_motivo(
        self, hourly_panel: Panel
    ) -> None:
        exigente = SeasonalNaiveProbe(
            model_id=ModelId("probe_exigente"),
            requires=ModelRequirements(min_context=1_000_000),
        )
        result = backtest(hourly_panel, [exigente], PLAN)

        assert result.model_runs["status"].tolist() == ["skipped"] * 3
        assert result.model_runs["error"].str.contains("min_context").all()
        assert result.forecasts.empty
        assert list(result.forecasts.columns) == [
            *FORECAST_KEY_COLUMNS,
            *[quantile_column(q) for q in QUANTILES],
        ]

    def test_el_cruce_de_cuantiles_se_repara_y_se_cuenta(self, hourly_panel: Panel) -> None:
        result = backtest(hourly_panel, [CrossedQuantileProbe()], PLAN)

        columns = [quantile_column(q) for q in QUANTILES]
        values = result.forecasts[columns].to_numpy()
        assert (np.diff(values, axis=1) >= 0).all()
        # Reparar en silencio ocultaria un diagnostico del modelo.
        assert (result.model_runs["quantile_crossings"] == len(result.forecasts) / 3).all()

    def test_un_modelo_que_predice_su_propio_cutoff_detiene_el_run(
        self, hourly_panel: Panel
    ) -> None:
        # La fuga no es un fallo de modelo que se registre y se siga: es un run
        # invalido, y por eso la excepcion sale del motor en lugar de acabar en
        # una fila con status="failed".
        with pytest.raises(CutoffViolation, match="anterior o igual a su cutoff"):
            backtest(hourly_panel, [CutoffViolatingProbe()], PLAN)


class TestContratoDelRun:
    def test_exige_al_menos_un_modelo(self, hourly_panel: Panel) -> None:
        with pytest.raises(ValueError, match="al menos un modelo"):
            backtest(hourly_panel, [], PLAN)

    def test_rechaza_model_id_repetido(self, hourly_panel: Panel) -> None:
        with pytest.raises(ValueError, match="model_id repetido"):
            backtest(hourly_panel, [SeasonalNaiveProbe(), SeasonalNaiveProbe()], PLAN)

    def test_aborta_si_un_modelo_necesita_exogenas_futuras_y_no_hay_proveedor(
        self, hourly_panel: Panel
    ) -> None:
        # Evaluarlo sin ellas lo dejaria compitiendo bajo condiciones distintas
        # de las declaradas, que es una comparacion mentirosa sin sintoma.
        with pytest.raises(MissingFutrExog, match="needs_futr_exog"):
            backtest(hourly_panel, [ScaledExogProbe()], PLAN)

    def test_el_plan_valida_su_rejilla_de_cuantiles(self) -> None:
        with pytest.raises(ValueError, match="estrictamente creciente"):
            BacktestPlan(h=24, n_windows=2, quantiles=(0.9, 0.1))
        with pytest.raises(ValueError, match="fuera de"):
            BacktestPlan(h=24, n_windows=2, quantiles=(0.0, 0.5))

    def test_el_plan_valida_la_particion_al_construirse(self) -> None:
        from chronolab.errors import WindowValidationError

        with pytest.raises(WindowValidationError, match="exige train_size"):
            BacktestPlan(h=24, n_windows=2, mode="sliding")

    def test_model_runs_registra_una_fila_por_modelo_y_ventana(self, hourly_panel: Panel) -> None:
        result = backtest(hourly_panel, [SeasonalNaiveProbe(), FailingProbe()], PLAN)
        assert len(result.model_runs) == 2 * 3
        assert result.model_runs["window_id"].dtype == np.int16
        assert result.model_runs["n_params"].dtype == "Int64"

    def test_el_plan_valida_la_politica_de_refit(self) -> None:
        with pytest.raises(ValueError, match="refit_every debe ser >= 1"):
            BacktestPlan(h=24, n_windows=2, refit_every=0)

    def test_aborta_si_el_panel_no_declara_exogenas_futuras(self, hourly_frame: Panel) -> None:
        sin_futuras = Panel(
            df=hourly_frame[["unique_id", "ds", "y", "voltage"]],
            spec=PanelSpec(
                dataset_id=DatasetId("sin_futuras"),
                freq="h",
                seasonalities=(24, 168),
                hist_exog=("voltage",),
            ),
        )
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=sin_futuras)

        with pytest.raises(MissingFutrExog, match="no declara ninguna columna futr_exog"):
            backtest(sin_futuras, [ScaledExogProbe()], PLAN, futr=provider)


class TestBarrerasDelMotor:
    """El motor no da por bueno lo que le entregan sus colaboradores."""

    def test_detiene_el_run_si_las_futuras_traen_la_objetivo(
        self, hourly_panel: Panel, realized_futr: RealizedFutrProvider
    ) -> None:
        provider = _ManipulatedProvider(inner=realized_futr, kind="target")
        with pytest.raises(LeakageError, match=r"\['y'\]"):
            backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN, futr=provider)

    def test_detiene_el_run_si_las_futuras_traen_una_historica(
        self, hourly_panel: Panel, realized_futr: RealizedFutrProvider
    ) -> None:
        # `voltage` solo se conoce hasta el cutoff. Que aparezca en el tramo de
        # prediccion es la fuga L7, y no hay forma de que sea inofensiva.
        provider = _ManipulatedProvider(inner=realized_futr, kind="hist")
        with pytest.raises(LeakageError, match="voltage"):
            backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN, futr=provider)

    def test_detiene_el_run_si_las_futuras_caen_antes_del_cutoff(
        self, hourly_panel: Panel, realized_futr: RealizedFutrProvider
    ) -> None:
        provider = _ManipulatedProvider(inner=realized_futr, kind="past")
        with pytest.raises(CutoffViolation, match="exogenas futuras"):
            backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN, futr=provider)

    def test_rechaza_una_trama_de_futuras_incompleta(
        self, hourly_panel: Panel, realized_futr: RealizedFutrProvider
    ) -> None:
        provider = _ManipulatedProvider(inner=realized_futr, kind="missing")
        with pytest.raises(ChronolabError, match="faltan exogenas futuras"):
            backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN, futr=provider)

    def test_rechaza_una_trama_de_futuras_de_otra_ventana(
        self, hourly_panel: Panel, realized_futr: RealizedFutrProvider
    ) -> None:
        provider = _ManipulatedProvider(inner=realized_futr, kind="otra_ventana")
        with pytest.raises(ChronolabError, match="ventana"):
            backtest(hourly_panel, [SeasonalNaiveProbe()], PLAN, futr=provider)

    def test_detiene_el_run_si_un_ajuste_dice_conocer_mas_de_lo_que_puede(
        self, hourly_panel: Panel
    ) -> None:
        # Reutilizar un ajuste antiguo es legitimo; usar uno que dice estar
        # ajustado mas alla del cutoff de la ventana no lo es en ningun caso.
        with pytest.raises(CutoffViolation, match="posterior al cutoff"):
            backtest(hourly_panel, [_LyingCutoffProbe()], PLAN)

    @pytest.mark.parametrize(
        ("kind", "motivo"),
        [
            ("sin_y_hat", "columnas obligatorias"),
            ("ds_no_temporal", "no es datetime"),
            ("ds_con_huso", "huso horario"),
            ("faltan_filas", "filas"),
            ("serie_de_mas", "series"),
            ("duplicadas", "duplicadas"),
            ("fuera_de_rejilla", "fuera de"),
        ],
    )
    def test_una_prediccion_que_incumple_el_contrato_es_un_fallo_del_modelo(
        self, hourly_panel: Panel, kind: str, motivo: str
    ) -> None:
        # Contrato incumplido no es fuga: se registra y el run continua, porque
        # el modelo roto tiene que verse en el leaderboard, no tumbar el run.
        result = backtest(hourly_panel, [_MalformedProbe(kind=kind)], PLAN)

        assert result.model_runs["status"].tolist() == ["failed"] * 3
        assert result.model_runs["error"].str.contains(motivo).all()
        assert result.forecasts.empty
