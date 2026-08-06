"""LSTM propio: red, bucle de entrenamiento y adaptador al protocolo `Forecaster`.

Redes minusculas y presupuestos de dos o tres epocas a proposito: lo que se
comprueba es que el escalado se invierte, que el early stopping para y
restaura, que los cuantiles no se cruzan y que el adaptador cumple el contrato
—no que el modelo prediga bien, para lo que esta el backtest del hito.

`torch` vive en el extra `deep` (D20), no en el nucleo: `pytest.importorskip`
a nivel de modulo salta el fichero entero con limpieza en el job `quality` de
CI, que hace `uv sync` a secas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from chronolab.data.futr import RealizedFutrProvider
from chronolab.errors import PerfectForesightWarning
from chronolab.evaluation.backtest import BacktestPlan, backtest
from chronolab.models.adapters.torch_lstm import LSTMForecaster
from chronolab.models.protocols import QUANTILES, FittedForecaster, Forecaster
from chronolab.models.torch.dataset import SeriesScaler, build_windows
from chronolab.models.torch.modules import build_lstm_net, count_parameters
from chronolab.models.torch.trainer import TrainConfig, seed_everything, train_lstm
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId

pytestmark = pytest.mark.slow

H = 6
INPUT_SIZE = 24
N_HOURS = 24 * 14

FAST = TrainConfig(max_epochs=3, batch_size=64, patience=2, val_fraction=0.2)


def _panel(n_series: int = 2) -> Panel:
    rng = np.random.default_rng(11)
    index = pd.date_range("2023-01-02", periods=N_HOURS, freq="h")
    temp = 12 + 8 * np.sin(2 * np.pi * (index.hour - 4) / 24) + rng.normal(0, 0.5, N_HOURS)
    parts = []
    for i in range(n_series):
        level = 50 + 5 * np.sin(2 * np.pi * np.arange(N_HOURS) / 24) + 10 * i
        y = level + 0.3 * np.abs(temp - 16) + rng.normal(0, 0.5, N_HOURS)
        parts.append(pd.DataFrame({"unique_id": f"s{i}", "ds": index, "y": y, "temp_c": temp}))
    spec = PanelSpec(
        dataset_id=DatasetId("mini"),
        freq="h",
        seasonalities=(24,),
        futr_exog=("temp_c",),
        tz_display="Europe/Madrid",
    )
    return Panel(df=pd.concat(parts, ignore_index=True), spec=spec)


@pytest.fixture(scope="module")
def panel() -> Panel:
    return _panel()


class TestRed:
    def test_la_salida_tiene_la_forma_del_horizonte_y_la_rejilla(self) -> None:
        import torch

        net = build_lstm_net(
            n_context_features=2, n_futr_features=1, h=H, n_quantiles=7, hidden_size=8, num_layers=1
        )
        out = net(torch.zeros(4, INPUT_SIZE, 2), torch.zeros(4, H, 1))
        assert tuple(out.shape) == (4, H, 7)

    def test_los_cuantiles_salen_ordenados_por_construccion(self) -> None:
        # No se reparan despues: la cabeza emite incrementos no negativos y los
        # acumula, asi que el cruce es imposible incluso con pesos aleatorios.
        import torch

        net = build_lstm_net(
            n_context_features=1, n_futr_features=0, h=H, n_quantiles=7, hidden_size=8, num_layers=1
        )
        out = net(torch.randn(16, INPUT_SIZE, 1), torch.zeros(16, H, 0))
        assert bool((torch.diff(out, dim=-1) >= 0).all())

    def test_sin_exogenas_futuras_la_cabeza_es_mas_pequena(self) -> None:
        comun = {"n_context_features": 1, "h": H, "n_quantiles": 7, "hidden_size": 8}
        sin_exog = count_parameters(build_lstm_net(n_futr_features=0, **comun))
        con_exog = count_parameters(build_lstm_net(n_futr_features=3, **comun))
        assert con_exog > sin_exog

    def test_count_parameters_cuenta_solo_los_entrenables(self) -> None:
        net = build_lstm_net(
            n_context_features=1, n_futr_features=0, h=H, n_quantiles=3, hidden_size=8, num_layers=1
        )
        total = count_parameters(net)
        assert total > 0

        for parameter in net.encoder.parameters():
            parameter.requires_grad_(False)
        assert count_parameters(net) < total

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_context_features": 0},
            {"h": 0},
            {"n_quantiles": 0},
            {"hidden_size": 0},
            {"num_layers": 0},
            {"dropout": 1.0},
        ],
    )
    def test_rechaza_dimensiones_incoherentes(self, kwargs: dict[str, object]) -> None:
        base = {
            "n_context_features": 1,
            "n_futr_features": 0,
            "h": H,
            "n_quantiles": 3,
            "hidden_size": 8,
        }
        with pytest.raises(ValueError):
            build_lstm_net(**{**base, **kwargs})  # type: ignore[arg-type]


class TestEntrenamiento:
    def _batch(self, panel: Panel):
        scaler = SeriesScaler.fit(panel, ("temp_c",))
        return scaler, build_windows(panel, scaler, input_size=INPUT_SIZE, h=H)

    def test_el_informe_declara_el_corte_de_validacion(self, panel: Panel) -> None:
        _, batch = self._batch(panel)
        net = build_lstm_net(
            n_context_features=2, n_futr_features=1, h=H, n_quantiles=7, hidden_size=8, num_layers=1
        )

        report = train_lstm(net, batch, quantiles=QUANTILES, config=FAST, seed=0)

        assert report.n_train_windows + report.n_val_windows == len(batch)
        assert report.n_val_windows == pytest.approx(len(batch) * 0.2, rel=0.05)
        assert report.epochs_run <= FAST.max_epochs
        assert len(report.train_losses) == report.epochs_run

    def test_el_early_stopping_para_antes_del_presupuesto(self, panel: Panel) -> None:
        _, batch = self._batch(panel)
        net = build_lstm_net(
            n_context_features=2, n_futr_features=1, h=H, n_quantiles=7, hidden_size=8, num_layers=1
        )
        # Tasa de aprendizaje nula: la validacion no mejora nunca, asi que la
        # paciencia se agota y para mucho antes de las 50 epocas.
        config = TrainConfig(max_epochs=50, batch_size=64, patience=2, learning_rate=1e-12)

        report = train_lstm(net, batch, quantiles=QUANTILES, config=config, seed=0)

        assert report.early_stopped
        assert report.epochs_run < config.max_epochs

    def test_restaura_los_mejores_pesos_no_los_ultimos(self, panel: Panel) -> None:
        import torch

        _, batch = self._batch(panel)
        net = build_lstm_net(
            n_context_features=2, n_futr_features=1, h=H, n_quantiles=7, hidden_size=8, num_layers=1
        )
        config = TrainConfig(max_epochs=6, batch_size=64, patience=6)

        report = train_lstm(net, batch, quantiles=QUANTILES, config=config, seed=0)

        # Los pesos que quedan en la red reproducen la mejor perdida de
        # validacion, no la de la ultima epoca.
        quantiles = torch.tensor(list(QUANTILES), dtype=torch.float32)
        n_train = report.n_train_windows
        with torch.no_grad():
            prediction = net(
                torch.from_numpy(batch.context[n_train:]), torch.from_numpy(batch.futr[n_train:])
            )
            target = torch.from_numpy(batch.target[n_train:])
            difference = target.unsqueeze(-1) - prediction
            actual = float(
                torch.maximum(quantiles * difference, (quantiles - 1.0) * difference).mean()
            )
        assert actual == pytest.approx(report.best_val_loss, rel=1e-4)

    def test_la_semilla_hace_el_entrenamiento_reproducible(self, panel: Panel) -> None:
        _, batch = self._batch(panel)
        losses = []
        for _ in range(2):
            seed_everything(3)
            net = build_lstm_net(
                n_context_features=2,
                n_futr_features=1,
                h=H,
                n_quantiles=7,
                hidden_size=8,
                num_layers=1,
            )
            losses.append(
                train_lstm(net, batch, quantiles=QUANTILES, config=FAST, seed=3).val_losses
            )
        assert losses[0] == pytest.approx(losses[1], rel=1e-6)

    def test_sin_ventanas_falla_con_un_mensaje_util(self, panel: Panel) -> None:
        scaler = SeriesScaler.fit(panel)
        vacio = build_windows(panel, scaler, input_size=N_HOURS * 2, h=H)
        net = build_lstm_net(
            n_context_features=1, n_futr_features=0, h=H, n_quantiles=7, hidden_size=8, num_layers=1
        )
        with pytest.raises(ValueError, match="ninguna ventana"):
            train_lstm(net, vacio, quantiles=QUANTILES, config=FAST, seed=0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"max_epochs": 0}, {"batch_size": 0}, {"patience": 0}, {"val_fraction": 0.0}],
    )
    def test_rechaza_presupuestos_incoherentes(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            TrainConfig(**kwargs)  # type: ignore[arg-type]


class TestAdaptador:
    def test_satisface_forecaster_y_fitted_forecaster(self, panel: Panel) -> None:
        model = LSTMForecaster(input_size=INPUT_SIZE, hidden_size=8, num_layers=1, config=FAST)
        assert isinstance(model, Forecaster)

        fitted = model.fit(panel, h=H)

        assert isinstance(fitted, FittedForecaster)
        assert fitted.cutoff == panel.last_ds
        assert fitted.h == H
        assert fitted.fit_seconds > 0.0

    def test_n_params_es_un_numero_real(self, panel: Panel) -> None:
        fitted = LSTMForecaster(
            input_size=INPUT_SIZE, hidden_size=8, num_layers=1, config=FAST
        ).fit(panel, h=H)
        assert isinstance(fitted.n_params, int)
        assert fitted.n_params > 0

    def test_declara_que_necesita_exogenas_futuras(self) -> None:
        assert LSTMForecaster().requires.needs_futr_exog is True
        assert LSTMForecaster(use_futr_exog=False).requires.needs_futr_exog is False

    def test_no_soporta_recursividad(self) -> None:
        # La cabeza proyecta el horizonte completo de una vez: no hay
        # realimentacion que declarar.
        assert LSTMForecaster().requires.supports_recursive is False

    def test_la_prediccion_vuelve_a_la_escala_original(self, panel: Panel) -> None:
        model = LSTMForecaster(input_size=INPUT_SIZE, hidden_size=8, num_layers=1, config=FAST)
        fitted = model.fit(panel, h=H)
        futr = _futr_frame(panel, fitted.h)

        prediction = fitted.predict(futr)

        # El panel vive en torno a 50-60; una prediccion en escala
        # estandarizada estaria en torno a cero y este test lo delataria.
        observed = panel.df["y"].to_numpy()
        assert prediction["y_hat"].between(observed.min() * 0.5, observed.max() * 1.5).all()

    def test_los_cuantiles_salen_ordenados_y_la_mediana_es_el_punto(self, panel: Panel) -> None:
        fitted = LSTMForecaster(
            input_size=INPUT_SIZE, hidden_size=8, num_layers=1, config=FAST
        ).fit(panel, h=H)

        prediction = fitted.predict(_futr_frame(panel, fitted.h))

        columns = [f"q_{round(q * 10000):04d}" for q in QUANTILES]
        values = prediction[columns].to_numpy()
        assert (np.diff(values, axis=1) >= -1e-6).all()
        np.testing.assert_allclose(prediction["y_hat"], prediction["q_5000"], atol=1e-9)

    def test_sin_exogenas_futuras_cuando_hacen_falta_lanza(self, panel: Panel) -> None:
        from chronolab.errors import MissingFutrExog

        fitted = LSTMForecaster(
            input_size=INPUT_SIZE, hidden_size=8, num_layers=1, config=FAST
        ).fit(panel, h=H)
        with pytest.raises(MissingFutrExog):
            fitted.predict(None)

    def test_un_tramo_demasiado_corto_falla_al_ajustar(self) -> None:
        corto = _panel(n_series=1).slice(
            pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-02 05:00")
        )
        model = LSTMForecaster(input_size=INPUT_SIZE, hidden_size=8, num_layers=1, config=FAST)
        with pytest.raises(ValueError, match="ninguna ventana"):
            model.fit(corto, h=H)

    @pytest.mark.parametrize(
        "kwargs", [{"input_size": 0}, {"quantiles": (0.5, 0.1)}, {"quantiles": (1.5,)}]
    )
    def test_rechaza_configuraciones_invalidas(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            LSTMForecaster(**kwargs)  # type: ignore[arg-type]


class TestIntegracionConElMotor:
    def test_un_backtest_con_refit_por_ventana_no_produce_fuga(self, panel: Panel) -> None:
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        plan = BacktestPlan(h=H, n_windows=2, step_size=H, refit_every=1)
        model = LSTMForecaster(input_size=INPUT_SIZE, hidden_size=8, num_layers=1, config=FAST)

        result = backtest(panel, [model], plan, futr=provider)

        assert (result.model_runs["status"] == "ok").all()
        assert (result.forecasts["ds"] > result.forecasts["cutoff"]).all()
        assert not result.forecasts["q_5000"].isna().any()

    def test_reutilizar_el_ajuste_falla_ruidosamente(self, panel: Panel) -> None:
        # `refit_cost="expensive"` hace que la politica por defecto sea un solo
        # ajuste por run; la cabeza solo sabe emitir los pasos 1..h desde su
        # propio cutoff, asi que la segunda ventana tiene que fallar en vez de
        # publicar una prediccion desfasada.
        with pytest.warns(PerfectForesightWarning):
            provider = RealizedFutrProvider(panel=panel)
        plan = BacktestPlan(h=H, n_windows=2, step_size=H)
        model = LSTMForecaster(input_size=INPUT_SIZE, hidden_size=8, num_layers=1, config=FAST)

        result = backtest(panel, [model], plan, futr=provider)

        assert result.model_runs["status"].tolist() == ["ok", "failed"]
        assert "refit_every=1" in result.model_runs["error"].dropna().iloc[0]

    def test_sin_futrprovider_el_run_aborta(self, panel: Panel) -> None:
        from chronolab.errors import MissingFutrExog

        plan = BacktestPlan(h=H, n_windows=1)
        with pytest.raises(MissingFutrExog):
            backtest(panel, [LSTMForecaster(config=FAST)], plan)


def _futr_frame(panel: Panel, h: int):
    """`FutrFrame` de las `h` horas siguientes al final del panel."""
    from chronolab.panel import FutrFrame
    from chronolab.types import Vintage

    grid = pd.date_range(panel.last_ds, periods=h + 1, freq="h")[1:]
    rng = np.random.default_rng(5)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": grid,
                    "temp_c": 12
                    + 8 * np.sin(2 * np.pi * (grid.hour - 4) / 24)
                    + rng.normal(0, 0.5, h),
                }
            )
            for uid in panel.ids()
        ],
        ignore_index=True,
    )
    return FutrFrame(df=frame, window=None, vintage=Vintage.REALIZED)  # type: ignore[arg-type]
