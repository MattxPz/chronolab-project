"""Rollup de `forecasts` al leaderboard, con la regla de no agregar agregados.

El run de juguete esta hecho a mano para que todas las cifras se puedan
comprobar sin ejecutar nada:

- serie ``s0``: entrenamiento ``[10, 20, 30, 40, 50, 62]`` -> ``q = 20.5``;
  evaluacion ``[70, 80]`` predicha **exactamente** -> error 0.
- serie ``s1``: entrenamiento ``[1, 2, 4, 8, 16, 32]`` -> ``q = 11.25``;
  evaluacion ``[64, 128]`` predicha con 10 de mas en los dos puntos.

De ahi: RMSE por serie 0 y 10, RMSE del modelo ``sqrt(200/4) = 7.0711``, y ese
par de numeros es justo el que demuestra que la fila agregada no puede salir de
promediar las de serie.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronolab.errors import UnstableMetricWarning
from chronolab.evaluation.aggregate import (
    DEFAULT_LEADERBOARD_PATH,
    build_leaderboard,
    score_forecasts,
    select_stage,
)
from chronolab.evaluation.backtest import BacktestPlan, BacktestResult, backtest
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId
from tests.fixtures.models import CrossedQuantileProbe, SeasonalNaiveProbe

START = pd.Timestamp("2023-01-02")
QUANTILES = (0.25, 0.5, 0.75)
PLAN = BacktestPlan(h=2, n_windows=1, quantiles=QUANTILES)

SERIES = {
    "s0": [10.0, 20.0, 30.0, 40.0, 50.0, 62.0, 70.0, 80.0],
    "s1": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
}
PREDICTIONS = {
    # (modelo, serie) -> predicciones de los dos instantes evaluados
    ("exacto", "s0"): [70.0, 80.0],
    ("exacto", "s1"): [74.0, 138.0],
    ("flojo", "s0"): [90.0, 100.0],
    ("flojo", "s1"): [84.0, 148.0],
}


def _panel() -> Panel:
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range(START, periods=len(values), freq="h"),
                    "y": values,
                }
            )
            for uid, values in SERIES.items()
        ],
        ignore_index=True,
    )
    spec = PanelSpec(dataset_id=DatasetId("mini"), freq="h", seasonalities=(2, 4))
    return Panel(df=frame, spec=spec)


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": np.array([0], dtype="int16"),
            "stage": ["holdout"],
            "train_start": [START],
            "cutoff": [START + pd.Timedelta(hours=5)],
            "first_pred": [START + pd.Timedelta(hours=6)],
            "last_pred": [START + pd.Timedelta(hours=7)],
            "n_train_obs": [12],
            "n_series": [2],
        }
    )


def _forecasts() -> pd.DataFrame:
    instantes = [START + pd.Timedelta(hours=6), START + pd.Timedelta(hours=7)]
    rows: list[pd.DataFrame] = []
    for (model_id, uid), predicted in PREDICTIONS.items():
        observed = SERIES[uid][6:]
        frame = pd.DataFrame(
            {
                "unique_id": uid,
                "model_id": model_id,
                "window_id": np.int16(0),
                "cutoff": START + pd.Timedelta(hours=5),
                "ds": instantes,
                "h_step": np.array([1, 2], dtype="int16"),
                "y": observed,
                "y_hat": predicted,
            }
        )
        # Solo el modelo "exacto" produce cuantiles; el otro deja NaN, como hace
        # el motor con los modelos sin soporte probabilistico.
        for quantile, offset in zip(QUANTILES, (-5.0, 0.0, 5.0), strict=True):
            column = f"q_{round(quantile * 10000):04d}"
            frame[column] = (
                [value + offset for value in predicted] if model_id == "exacto" else np.nan
            )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _model_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["exacto", "flojo"],
            "window_id": np.array([0, 0], dtype="int16"),
            "status": ["ok", "ok"],
            "error": [None, None],
            "refit": [True, True],
            "refit_every": np.array([1, 1], dtype="int16"),
            "fit_seconds": np.array([2.0, 0.5], dtype="float32"),
            "predict_seconds": np.array([0.25, 0.75], dtype="float32"),
            "n_params": pd.array([10, None], dtype="Int64"),
            "is_zero_shot": [False, True],
            "quantile_crossings": np.array([0, 0], dtype="int32"),
        }
    )


def _result() -> BacktestResult:
    return BacktestResult(
        forecasts=_forecasts(),
        windows=_windows(),
        model_runs=_model_runs(),
        spec=_panel().spec,
        plan=PLAN,
        futr_vintage=None,
    )


@pytest.fixture
def leaderboard() -> pd.DataFrame:
    return build_leaderboard(_result(), _panel(), quantiles=QUANTILES, path=None)


class TestSeleccionDeEtapa:
    def test_filtra_por_ventanas_de_la_etapa(self) -> None:
        result = _result()
        assert len(select_stage(result.forecasts, result.windows, "holdout")) == 8
        assert select_stage(result.forecasts, result.windows, "dev").empty
        assert len(select_stage(result.forecasts, result.windows, None)) == 8

    def test_un_run_sin_holdout_no_tiene_nada_que_publicar(self) -> None:
        result = _result()
        tabla = build_leaderboard(result, _panel(), stage="dev", quantiles=QUANTILES, path=None)
        # Vacia, no un leaderboard silencioso sobre ventanas de desarrollo.
        assert tabla.empty


class TestDenominadoresUnidos:
    def test_cada_fila_lleva_el_denominador_de_su_serie_y_ventana(self) -> None:
        scored = score_forecasts(_result(), _panel())
        por_serie = scored.groupby("unique_id")["mase_denominator"].unique()

        # s0: |30-10|, |40-20|, |50-30|, |62-40| -> (20+20+20+22)/4 = 20.5
        assert por_serie["s0"].tolist() == pytest.approx([20.5])
        # s1: |4-1|, |8-2|, |16-4|, |32-8| -> (3+6+12+24)/4 = 11.25
        assert por_serie["s1"].tolist() == pytest.approx([11.25])


class TestEsquemaDelLeaderboard:
    def test_una_fila_por_modelo_y_serie_mas_el_agregado(self, leaderboard: pd.DataFrame) -> None:
        assert len(leaderboard) == 2 * 2 + 2  # dos modelos x dos series + dos agregados
        assert leaderboard["unique_id"].isna().sum() == 2
        assert set(leaderboard["model_id"]) == {"exacto", "flojo"}

    def test_las_columnas_declaradas_estan_todas(self, leaderboard: pd.DataFrame) -> None:
        esperadas = {
            "model_id",
            "unique_id",
            "stage",
            "n_windows",
            "n_obs",
            "mae",
            "rmse",
            "mape",
            "smape",
            "mase",
            "mase_denominator",
            "pinball_mean",
            "crps_discrete",
            "coverage_50",
            "width_50",
            "n_params",
            "fit_seconds_total",
            "fit_seconds_mean",
            "predict_seconds_total",
            "predict_seconds_mean",
            "n_refits",
            "n_windows_ok",
            "n_windows_failed",
            "n_windows_skipped",
            "is_zero_shot",
        }
        assert esperadas <= set(leaderboard.columns)
        assert list(leaderboard.columns[:5]) == [
            "model_id",
            "unique_id",
            "stage",
            "n_windows",
            "n_obs",
        ]

    def test_las_metricas_por_serie_son_las_calculadas_a_mano(
        self, leaderboard: pd.DataFrame
    ) -> None:
        exacto = leaderboard.set_index(["model_id", "unique_id"]).loc["exacto"]

        # s0 se predice exactamente.
        assert exacto.loc["s0", "mae"] == pytest.approx(0.0)
        assert exacto.loc["s0", "mase"] == pytest.approx(0.0)
        # s1 falla por 10 en los dos puntos, con q = 11.25.
        assert exacto.loc["s1", "mae"] == pytest.approx(10.0)
        assert exacto.loc["s1", "rmse"] == pytest.approx(10.0)
        assert exacto.loc["s1", "mase"] == pytest.approx(10.0 / 11.25)
        # MAPE de s1: (10/64 + 10/128) / 2 = 11.71875 %
        assert exacto.loc["s1", "mape"] == pytest.approx(100 * (10 / 64 + 10 / 128) / 2)

    def test_el_agregado_no_es_el_promedio_de_las_series(self, leaderboard: pd.DataFrame) -> None:
        # La regla de oro del modulo. RMSE lo hace evidente: la raiz de la media
        # de los cuadrados no es la media de las raices.
        exacto = leaderboard[leaderboard["model_id"] == "exacto"]
        agregado = exacto[exacto["unique_id"].isna()].iloc[0]
        por_serie = exacto[exacto["unique_id"].notna()]["rmse"]

        assert agregado["rmse"] == pytest.approx(math.sqrt(200.0 / 4))
        assert agregado["rmse"] == pytest.approx(7.0710678, abs=1e-6)
        assert por_serie.mean() == pytest.approx(5.0)
        assert agregado["rmse"] != pytest.approx(por_serie.mean())

    def test_el_agregado_sale_de_las_filas_crudas(self, leaderboard: pd.DataFrame) -> None:
        scored = score_forecasts(_result(), _panel())
        crudo = scored[scored["model_id"] == "exacto"]
        errores = (crudo["y"] - crudo["y_hat"]).abs()

        agregado = leaderboard[
            (leaderboard["model_id"] == "exacto") & leaderboard["unique_id"].isna()
        ].iloc[0]
        assert agregado["mae"] == pytest.approx(errores.mean())
        assert agregado["mase"] == pytest.approx((errores / crudo["mase_denominator"]).mean())
        assert agregado["n_obs"] == 4

    def test_las_probabilisticas_solo_las_tiene_quien_produce_cuantiles(
        self, leaderboard: pd.DataFrame
    ) -> None:
        indexado = leaderboard.set_index(["model_id", "unique_id"], drop=False)
        exacto = indexado[indexado["model_id"] == "exacto"]
        flojo = indexado[indexado["model_id"] == "flojo"]

        # El intervalo 50 % es y_hat +-5; en s0 la prediccion es exacta, asi que
        # el observado cae dentro y la anchura es 10.
        assert exacto[exacto["unique_id"] == "s0"]["coverage_50"].item() == pytest.approx(1.0)
        assert exacto[exacto["unique_id"] == "s0"]["width_50"].item() == pytest.approx(10.0)
        # En s1 la prediccion se pasa por 10: el observado queda fuera.
        assert exacto[exacto["unique_id"] == "s1"]["coverage_50"].item() == pytest.approx(0.0)
        # Un modelo sin cuantiles no puntua: NaN, nunca cero.
        assert flojo["coverage_50"].isna().all()
        assert flojo["crps_discrete"].isna().all()

    def test_el_orden_pone_los_agregados_arriba_y_el_mejor_mase_primero(
        self, leaderboard: pd.DataFrame
    ) -> None:
        cabecera = leaderboard.head(2)
        assert cabecera["unique_id"].isna().all()
        assert cabecera["model_id"].tolist() == ["exacto", "flojo"]
        assert cabecera["mase"].is_monotonic_increasing

    def test_la_etapa_evaluada_viaja_en_cada_fila(self) -> None:
        todas = build_leaderboard(_result(), _panel(), quantiles=QUANTILES, path=None)
        holdout = build_leaderboard(
            _result(), _panel(), stage="holdout", quantiles=QUANTILES, path=None
        )
        assert (todas["stage"] == "all").all()
        assert (holdout["stage"] == "holdout").all()

    def test_se_puede_prescindir_del_agregado(self) -> None:
        tabla = build_leaderboard(
            _result(), _panel(), quantiles=QUANTILES, include_overall=False, path=None
        )
        assert tabla["unique_id"].notna().all()
        assert len(tabla) == 4


class TestAvisoDeMapeInestable:
    def test_resume_en_un_solo_aviso_las_combinaciones_afectadas(self) -> None:
        # Un leaderboard de doce modelos por diez series emitiria ciento veinte
        # avisos identicos si cada grupo avisara por su cuenta. Aqui sale uno,
        # con la lista de los afectados.
        result = _result()
        forecasts = result.forecasts.copy()
        cero = (forecasts["unique_id"] == "s1") & (forecasts["h_step"] == 1)
        forecasts.loc[cero, "y"] = 0.0
        con_cero = BacktestResult(
            forecasts=forecasts,
            windows=result.windows,
            model_runs=result.model_runs,
            spec=result.spec,
            plan=result.plan,
            futr_vintage=None,
        )

        with pytest.warns(UnstableMetricWarning, match="MAPE inestable") as avisos:
            tabla = build_leaderboard(con_cero, _panel(), quantiles=QUANTILES, path=None)

        assert len(avisos) == 1
        assert "Ordena el leaderboard por MASE" in str(avisos[0].message)
        # Y el numero se sigue publicando: el aviso informa, no censura.
        assert tabla["mape"].notna().any()

    def test_una_serie_lejos_de_cero_no_avisa(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UnstableMetricWarning)
            build_leaderboard(_result(), _panel(), quantiles=QUANTILES, path=None)


class TestCoste:
    def test_los_tiempos_vienen_de_model_runs(self, leaderboard: pd.DataFrame) -> None:
        exacto = leaderboard[leaderboard["model_id"] == "exacto"]
        flojo = leaderboard[leaderboard["model_id"] == "flojo"]

        assert exacto["fit_seconds_total"].tolist() == pytest.approx([2.0] * len(exacto))
        assert exacto["predict_seconds_total"].tolist() == pytest.approx([0.25] * len(exacto))
        assert exacto["fit_seconds_mean"].tolist() == pytest.approx([2.0] * len(exacto))
        assert flojo["fit_seconds_total"].tolist() == pytest.approx([0.5] * len(flojo))
        assert flojo["is_zero_shot"].all()
        assert not exacto["is_zero_shot"].any()

    def test_el_coste_se_repite_en_las_filas_de_la_misma_modelo(
        self, leaderboard: pd.DataFrame
    ) -> None:
        # Un ajuste cubre todas las series a la vez: el coste es del par
        # (modelo, ventana) y no se puede repartir entre series.
        por_modelo = leaderboard.groupby("model_id")["fit_seconds_total"].nunique()
        assert (por_modelo == 1).all()

    def test_cuenta_las_ventanas_por_estado(self, leaderboard: pd.DataFrame) -> None:
        assert (leaderboard["n_windows_ok"] == 1).all()
        assert (leaderboard["n_windows_failed"] == 0).all()
        assert (leaderboard["n_refits"] == 1).all()

    def test_el_tamano_del_modelo_viaja_junto_al_coste(self, leaderboard: pd.DataFrame) -> None:
        # Sin `n_params`, un LSTM de cinco mil parametros y un transformer de
        # medio millon se leen igual en una tabla ordenada por MASE.
        exacto = leaderboard[leaderboard["model_id"] == "exacto"]
        assert (exacto["n_params"] == 10).all()

    def test_un_modelo_sin_parametros_ajustados_queda_nulo_no_cero(
        self, leaderboard: pd.DataFrame
    ) -> None:
        # "No tiene parametros que ajustar" (baselines, statsforecast) es
        # distinto de "tiene cero parametros", y el leaderboard tiene que poder
        # distinguirlo.
        flojo = leaderboard[leaderboard["model_id"] == "flojo"]
        assert flojo["n_params"].isna().all()

    def test_n_params_sobrevive_al_parquet_como_entero_anulable(self, tmp_path: Path) -> None:
        destino = tmp_path / "leaderboard.parquet"
        build_leaderboard(_result(), _panel(), quantiles=QUANTILES, path=destino)

        releido = pd.read_parquet(destino)

        assert releido["n_params"].dtype == "Int64"
        assert releido.loc[releido["model_id"] == "exacto", "n_params"].eq(10).all()
        assert releido.loc[releido["model_id"] == "flojo", "n_params"].isna().all()


class TestPersistencia:
    def test_escribe_el_parquet_y_se_relee_igual(self, tmp_path: Path) -> None:
        destino = tmp_path / "resultados" / "leaderboard.parquet"
        tabla = build_leaderboard(_result(), _panel(), quantiles=QUANTILES, path=destino)

        assert destino.is_file()
        releido = pd.read_parquet(destino)
        pd.testing.assert_frame_equal(releido, tabla)

    def test_no_deja_ficheros_temporales(self, tmp_path: Path) -> None:
        destino = tmp_path / "leaderboard.parquet"
        build_leaderboard(_result(), _panel(), quantiles=QUANTILES, path=destino)
        assert [p.name for p in tmp_path.iterdir()] == ["leaderboard.parquet"]

    def test_sobrescribe_de_forma_atomica(self, tmp_path: Path) -> None:
        destino = tmp_path / "leaderboard.parquet"
        build_leaderboard(_result(), _panel(), quantiles=QUANTILES, path=destino)
        primera = pd.read_parquet(destino)

        build_leaderboard(
            _result(), _panel(), quantiles=QUANTILES, include_overall=False, path=destino
        )
        segunda = pd.read_parquet(destino)
        assert len(segunda) < len(primera)

    def test_sin_ruta_no_escribe_nada(self, tmp_path: Path) -> None:
        build_leaderboard(_result(), _panel(), quantiles=QUANTILES, path=None)
        assert list(tmp_path.iterdir()) == []

    def test_la_ruta_por_defecto_es_la_del_proyecto(self) -> None:
        assert str(DEFAULT_LEADERBOARD_PATH) == str(Path("reports/results/leaderboard.parquet"))
        assert DEFAULT_LEADERBOARD_PATH.parent.name == "results"


class TestExtremoAExtremo:
    def test_agrega_un_run_real_del_motor(self, hourly_panel: Panel, tmp_path: Path) -> None:
        plan = BacktestPlan(h=24, n_windows=3, step_size=24, holdout_windows=2)
        resultado = backtest(hourly_panel, [SeasonalNaiveProbe(), CrossedQuantileProbe()], plan)

        tabla = build_leaderboard(
            resultado, hourly_panel, stage="holdout", path=tmp_path / "lb.parquet"
        )

        assert set(tabla["model_id"]) == {"probe_seasonal_naive", "probe_crossed_quantiles"}
        assert (tabla["stage"] == "holdout").all()
        assert (tabla["n_windows"] == 2).all()
        # 3 series x 24 pasos x 2 ventanas de holdout
        agregados = tabla[tabla["unique_id"].isna()]
        assert (agregados["n_obs"] == 3 * 24 * 2).all()
        # El naive estacional sobre una serie estacional deberia rondar el 1.
        naive = agregados[agregados["model_id"] == "probe_seasonal_naive"]["mase"].item()
        assert 0.1 < naive < 3.0
