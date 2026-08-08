"""Tests de `chronolab.viz.plots`.

Los `compute_*` se prueban a fondo (son numericos, se pueden verificar
exactamente); los `plot_*` con pruebas de humo y de estructura (devuelven una
`Figure`, con las trazas y los datos correctos), no de estilo visual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from chronolab.viz import plots
from tests.fixtures.synthetic import make_hourly_frame

# --------------------------------------------------------------------------- #
# Fixtures compartidas
# --------------------------------------------------------------------------- #


@pytest.fixture
def seasonal_series() -> pd.Series:
    """Una serie con estacionalidad diaria y semanal conocida, sin huecos."""
    frame = make_hourly_frame(n_series=1, n_hours=168 * 10, seed=0)
    return frame.set_index("ds")["y"]


@pytest.fixture
def white_noise_series() -> pd.Series:
    """Ruido blanco puro: sin estructura que un compute_* pudiera "encontrar" por error."""
    index = pd.date_range("2023-01-01", periods=24 * 90, freq="h")
    values = np.random.default_rng(0).normal(0.0, 1.0, len(index))
    return pd.Series(values, index=index)


# --------------------------------------------------------------------------- #
# series_color_map
# --------------------------------------------------------------------------- #


class TestSeriesColorMap:
    def test_asigna_en_orden(self) -> None:
        mapping = plots.series_color_map(["a", "b", "c"])
        assert mapping["a"] == plots.CATEGORICAL[0]
        assert mapping["b"] == plots.CATEGORICAL[1]
        assert mapping["c"] == plots.CATEGORICAL[2]

    def test_mismo_orden_da_el_mismo_mapa(self) -> None:
        assert plots.series_color_map(["x", "y"]) == plots.series_color_map(["x", "y"])

    def test_mas_de_ocho_series_no_revienta(self) -> None:
        mapping = plots.series_color_map([f"s{i}" for i in range(12)])
        assert len(mapping) == 12
        assert mapping["s7"] == plots.CATEGORICAL[7]
        assert mapping["s11"] == plots.CATEGORICAL[7]  # repite el ultimo


# --------------------------------------------------------------------------- #
# 2. MSTL
# --------------------------------------------------------------------------- #


class TestComputeMstl:
    def test_columnas_esperadas(self, seasonal_series: pd.Series) -> None:
        components = plots.compute_mstl(seasonal_series, periods=(24, 168))
        for column in ("observed", "trend", "seasonal_24", "seasonal_168", "resid"):
            assert column in components.columns

    def test_reconstruye_la_serie_observada(self, seasonal_series: pd.Series) -> None:
        # La descomposicion aditiva debe sumar de vuelta al observado.
        components = plots.compute_mstl(seasonal_series, periods=(24, 168))
        reconstructed = (
            components["trend"]
            + components["seasonal_24"]
            + components["seasonal_168"]
            + components["resid"]
        )
        np.testing.assert_allclose(
            reconstructed.to_numpy(), components["observed"].to_numpy(), atol=1e-6
        )

    def test_interpola_los_huecos_antes_de_descomponer(self) -> None:
        index = pd.date_range("2023-01-01", periods=24 * 20, freq="h")
        values = np.sin(2 * np.pi * np.arange(len(index)) / 24)
        series = pd.Series(values, index=index)
        series.iloc[50] = np.nan
        components = plots.compute_mstl(series, periods=(24,))
        assert not components["observed"].isna().any()

    def test_un_solo_periodo_tambien_funciona(self, seasonal_series: pd.Series) -> None:
        components = plots.compute_mstl(seasonal_series, periods=(24,))
        assert "seasonal_24" in components.columns
        assert "seasonal_168" not in components.columns


class TestPlotMstl:
    def test_devuelve_una_figura_con_un_panel_por_componente(
        self, seasonal_series: pd.Series
    ) -> None:
        components = plots.compute_mstl(seasonal_series, periods=(24, 168))
        fig = plots.plot_mstl(components, periods=(24, 168))
        assert isinstance(fig, go.Figure)
        # observado + tendencia + 2 estacionalidades + residuo = 5 trazas.
        assert len(fig.data) == 5


# --------------------------------------------------------------------------- #
# 3. ACF, PACF, periodograma
# --------------------------------------------------------------------------- #


class TestComputeAcfPacf:
    def test_el_retardo_cero_de_acf_es_siempre_uno(self, white_noise_series: pd.Series) -> None:
        table = plots.compute_acf_pacf(white_noise_series, max_lag=50)
        assert table["acf"].iloc[0] == pytest.approx(1.0)

    def test_ruido_blanco_no_tiene_autocorrelacion_fuerte(
        self, white_noise_series: pd.Series
    ) -> None:
        table = plots.compute_acf_pacf(white_noise_series, max_lag=50)
        # Con ruido blanco, salvo el propio lag 0, la autocorrelacion deberia
        # quedarse dentro de una banda estrecha con alta probabilidad.
        assert table["acf"].iloc[1:].abs().max() < 0.3

    def test_una_sinusoide_pura_tiene_autocorrelacion_fuerte_en_su_periodo(self) -> None:
        index = pd.date_range("2023-01-01", periods=24 * 30, freq="h")
        series = pd.Series(np.sin(2 * np.pi * np.arange(len(index)) / 24), index=index)
        table = plots.compute_acf_pacf(series, max_lag=48)
        assert table.loc[table["lag"] == 24, "acf"].iloc[0] > 0.9

    def test_longitud_es_max_lag_mas_uno(self, white_noise_series: pd.Series) -> None:
        table = plots.compute_acf_pacf(white_noise_series, max_lag=30)
        assert len(table) == 31


class TestPlotAcfPacf:
    def test_devuelve_dos_paneles_de_barras(self, white_noise_series: pd.Series) -> None:
        table = plots.compute_acf_pacf(white_noise_series, max_lag=50)
        fig = plots.plot_acf_pacf(table, n_obs=len(white_noise_series))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert all(isinstance(trace, go.Bar) for trace in fig.data)


class TestComputePeriodogram:
    def test_detecta_el_periodo_de_una_sinusoide_pura(self) -> None:
        index = pd.date_range("2023-01-01", periods=24 * 60, freq="h")
        series = pd.Series(np.sin(2 * np.pi * np.arange(len(index)) / 24), index=index)
        table = plots.compute_periodogram(series)
        dominant_period = table.loc[table["power"].idxmax(), "period"]
        assert dominant_period == pytest.approx(24.0, rel=0.05)

    def test_no_incluye_la_componente_de_frecuencia_cero(self) -> None:
        table = plots.compute_periodogram(pd.Series(np.arange(100, dtype=float)))
        assert (table["frequency"] > 0).all()


class TestPlotPeriodogram:
    def test_devuelve_una_figura_con_las_lineas_de_referencia(self) -> None:
        index = pd.date_range("2023-01-01", periods=24 * 60, freq="h")
        series = pd.Series(np.sin(2 * np.pi * np.arange(len(index)) / 24), index=index)
        table = plots.compute_periodogram(series)
        fig = plots.plot_periodogram(table, highlight_periods=(24.0, 168.0))
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.shapes) == 2  # las dos vlines


# --------------------------------------------------------------------------- #
# 4. Perfiles agregados
# --------------------------------------------------------------------------- #


class TestComputeHourDowMatrix:
    def test_valores_coinciden_con_el_calculo_manual(self) -> None:
        frame = pd.DataFrame({"hour": [0, 0, 1], "dayofweek": [0, 0, 0], "y": [10.0, 20.0, 5.0]})
        matrix = plots.compute_hour_dow_matrix(frame)
        assert matrix.loc[0, "Lun"] == pytest.approx(15.0)
        assert matrix.loc[1, "Lun"] == pytest.approx(5.0)

    def test_forma_24_por_7(self) -> None:
        frame = pd.DataFrame({"hour": [5], "dayofweek": [2], "y": [1.0]})
        matrix = plots.compute_hour_dow_matrix(frame)
        assert matrix.shape == (24, 7)
        assert list(matrix.columns) == ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]


class TestPlotHourDowHeatmap:
    def test_devuelve_un_heatmap_con_la_forma_correcta(self) -> None:
        frame = pd.DataFrame({"hour": [5], "dayofweek": [2], "y": [1.0]})
        matrix = plots.compute_hour_dow_matrix(frame)
        fig = plots.plot_hour_dow_heatmap(matrix)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)
        assert np.array(fig.data[0].z).shape == (24, 7)


class TestComputeMonthlyProfile:
    def test_media_y_cuantiles_por_mes(self) -> None:
        frame = pd.DataFrame({"month": [1, 1, 1, 6], "y": [1.0, 2.0, 3.0, 100.0]})
        table = plots.compute_monthly_profile(frame)
        row_jan = table[table["month"] == 1].iloc[0]
        assert row_jan["mean"] == pytest.approx(2.0)

    def test_incluye_los_doce_meses_aunque_falten_datos(self) -> None:
        frame = pd.DataFrame({"month": [1], "y": [1.0]})
        table = plots.compute_monthly_profile(frame)
        assert len(table) == 12
        assert table["month"].tolist() == list(range(1, 13))


class TestPlotMonthlyProfile:
    def test_tres_trazas_banda_y_media(self) -> None:
        frame = pd.DataFrame({"month": list(range(1, 13)), "y": list(range(12))})
        table = plots.compute_monthly_profile(frame)
        fig = plots.plot_monthly_profile(table)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3


class TestPlotHolidayEffect:
    def test_dos_cajas_holiday_y_no_holiday(self) -> None:
        frame = pd.DataFrame(
            {"is_holiday": [False, False, False, True, True], "y": [10.0, 11.0, 9.0, 5.0, 4.0]}
        )
        fig = plots.plot_holiday_effect(frame)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert all(isinstance(trace, go.Box) for trace in fig.data)


# --------------------------------------------------------------------------- #
# 5. Relacion demanda-temperatura
# --------------------------------------------------------------------------- #


class TestComputeLowessFit:
    def test_sigue_una_relacion_monotona(self) -> None:
        rng = np.random.default_rng(0)
        x = pd.Series(np.linspace(0, 10, 200))
        y = pd.Series(2.0 * x + rng.normal(0, 0.2, 200))
        fit = plots.compute_lowess_fit(x, y, frac=0.3)
        assert fit["y_fit"].is_monotonic_increasing

    def test_descarta_nan(self) -> None:
        x = pd.Series([1.0, 2.0, np.nan, 4.0])
        y = pd.Series([1.0, 2.0, 3.0, np.nan])
        fit = plots.compute_lowess_fit(x, y, frac=0.5)
        assert len(fit) == 2


class TestPlotTemperatureScatter:
    def test_dos_trazas_puntos_y_ajuste(self) -> None:
        frame = pd.DataFrame({"temp_c": [10.0, 15.0, 20.0], "y": [100.0, 90.0, 110.0]})
        fit = plots.compute_lowess_fit(frame["temp_c"], frame["y"], frac=0.9)
        fig = plots.plot_temperature_scatter(frame, fit, color=plots.CATEGORICAL[0])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


class TestComputeDegreeDays:
    def test_hdd_y_cdd_por_encima_y_por_debajo_de_la_base(self) -> None:
        hours = pd.date_range("2024-01-01", periods=24, freq="h")
        frame = pd.DataFrame({"ds": hours, "temp_c": [5.0] * 24})  # frio todo el dia
        table = plots.compute_degree_days(frame, base_heating=18.0, base_cooling=22.0)
        assert table["hdd"].iloc[0] == pytest.approx(13.0)
        assert table["cdd"].iloc[0] == pytest.approx(0.0)

    def test_temperatura_en_zona_de_confort_no_genera_grados_dia(self) -> None:
        hours = pd.date_range("2024-01-01", periods=24, freq="h")
        frame = pd.DataFrame({"ds": hours, "temp_c": [20.0] * 24})
        table = plots.compute_degree_days(frame, base_heating=18.0, base_cooling=22.0)
        assert table["hdd"].iloc[0] == pytest.approx(0.0)
        assert table["cdd"].iloc[0] == pytest.approx(0.0)


class TestComputeResampledMean:
    def test_media_diaria(self) -> None:
        hours = pd.date_range("2024-01-01", periods=48, freq="h")
        frame = pd.DataFrame({"ds": hours, "y": [10.0] * 24 + [20.0] * 24})
        table = plots.compute_resampled_mean(frame, freq="D")
        assert table["y_mean"].tolist() == [10.0, 20.0]


class TestPlotDegreeDaysCorrelation:
    def test_la_barra_refleja_la_correlacion_calculada(self) -> None:
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        rng = np.random.default_rng(0)
        hdd = rng.uniform(0, 15, 30)
        cdd = rng.uniform(0, 0.01, 30)  # ~constante mas un ruido minimo: evita corr indefinida
        degree_days = pd.DataFrame({"date": dates, "hdd": hdd, "cdd": cdd, "temp_mean": 10.0})
        demand = pd.DataFrame({"date": dates, "y_mean": hdd * 3.0 + rng.normal(0, 0.1, 30)})

        expected_corr = np.corrcoef(hdd, demand["y_mean"])[0, 1]
        fig = plots.plot_degree_days_correlation(degree_days, demand)
        assert fig.data[0].y[0] == pytest.approx(expected_corr, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. Estadisticos de dificultad
# --------------------------------------------------------------------------- #


class TestComputeSeriesDifficulty:
    def test_una_serie_muy_estacional_tiene_fuerza_estacional_alta(
        self, seasonal_series: pd.Series
    ) -> None:
        stats = plots.compute_series_difficulty(seasonal_series, periods=(24, 168))
        assert stats["seasonal_strength_24"] > 0.5

    def test_ruido_blanco_tiene_fuerza_estacional_baja(self, white_noise_series: pd.Series) -> None:
        stats = plots.compute_series_difficulty(white_noise_series, periods=(24,))
        assert stats["seasonal_strength_24"] < 0.3

    def test_ruido_blanco_tiene_entropia_espectral_mayor_que_una_serie_estacional(
        self, seasonal_series: pd.Series, white_noise_series: pd.Series
    ) -> None:
        # Un espectro plano (ruido) reparte la potencia entre mas frecuencias
        # que un espectro concentrado en los picos estacionales: mayor entropia.
        seasonal_stats = plots.compute_series_difficulty(seasonal_series, periods=(24,))
        noise_stats = plots.compute_series_difficulty(white_noise_series, periods=(24,))
        assert noise_stats["spectral_entropy"] > seasonal_stats["spectral_entropy"]

    def test_todas_las_claves_esperadas(self, seasonal_series: pd.Series) -> None:
        stats = plots.compute_series_difficulty(seasonal_series, periods=(24, 168))
        for key in (
            "trend_strength",
            "seasonal_strength_24",
            "seasonal_strength_168",
            "spectral_entropy",
        ):
            assert key in stats
            assert 0.0 <= stats[key] <= 1.0 or key == "spectral_entropy"


class TestComputeDifficultyTable:
    def test_una_fila_por_serie(
        self, seasonal_series: pd.Series, white_noise_series: pd.Series
    ) -> None:
        table = plots.compute_difficulty_table(
            {"estacional": seasonal_series, "ruido": white_noise_series}, periods=(24,)
        )
        assert set(table["unique_id"]) == {"estacional", "ruido"}
        assert len(table) == 2


class TestPlotDifficultyTable:
    def test_devuelve_una_tabla_con_una_fila_por_serie(self) -> None:
        table = pd.DataFrame(
            {
                "unique_id": ["a", "b"],
                "trend_strength": [0.5, 0.1],
                "seasonal_strength_24": [0.9, 0.2],
                "spectral_entropy": [0.3, 0.8],
            }
        )
        fig = plots.plot_difficulty_table(table)
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Table)
        assert len(fig.data[0].cells.values[0]) == 2


# --------------------------------------------------------------------------- #
# 1. Perfil de calidad (figuras)
# --------------------------------------------------------------------------- #


class TestPlotQualityOverview:
    def test_cuatro_series_de_barras_con_los_valores_del_informe(self) -> None:
        report = pd.DataFrame(
            {
                "unique_id": ["a", "b"],
                "n_gaps": [1, 2],
                "n_duplicated_pairs": [0, 3],
                "n_zeros": [5, 0],
                "n_outliers": [2, 1],
            }
        )
        fig = plots.plot_quality_overview(report)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4
        assert list(fig.data[0].y) == [1, 2]


class TestPlotSeriesWithFlags:
    def test_una_traza_de_linea_y_una_de_atipicos(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 3,
                "ds": pd.date_range("2024-01-01", periods=3, freq="h"),
                "y": [1.0, np.nan, 3.0],
            }
        )
        outliers = pd.DataFrame(
            {"unique_id": ["a"], "ds": [pd.Timestamp("2024-01-01 02:00")], "y": [3.0]}
        )
        fig = plots.plot_series_with_flags(
            frame, outliers, unique_id="a", color=plots.CATEGORICAL[0]
        )
        assert len(fig.data) == 2

    def test_sin_atipicos_solo_hay_una_traza(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 2,
                "ds": pd.date_range("2024-01-01", periods=2, freq="h"),
                "y": [1.0, 2.0],
            }
        )
        empty_outliers = pd.DataFrame(columns=["unique_id", "ds", "y"])
        fig = plots.plot_series_with_flags(
            frame, empty_outliers, unique_id="a", color=plots.CATEGORICAL[0]
        )
        assert len(fig.data) == 1


class TestPlotDstContinuity:
    def test_devuelve_una_figura_acotada_a_la_ventana(self) -> None:
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 48,
                "ds": pd.date_range("2024-10-26", periods=48, freq="h"),
                "y": np.arange(48, dtype=float),
            }
        )
        fig = plots.plot_dst_continuity(
            frame,
            unique_id="a",
            transition=pd.Timestamp("2024-10-27"),
            window=pd.Timedelta(hours=4),
        )
        assert isinstance(fig, go.Figure)


# --------------------------------------------------------------------------- #
# 7. model_color_map
# --------------------------------------------------------------------------- #


class TestModelColorMap:
    def test_asigna_en_orden(self) -> None:
        mapping = plots.model_color_map(["naive", "mstl"])
        assert mapping["naive"] == plots.CATEGORICAL[0]
        assert mapping["mstl"] == plots.CATEGORICAL[1]

    def test_mismo_orden_da_el_mismo_mapa(self) -> None:
        assert plots.model_color_map(["a", "b"]) == plots.model_color_map(["a", "b"])


# --------------------------------------------------------------------------- #
# 8. Forecast: overlay y residuos
# --------------------------------------------------------------------------- #


@pytest.fixture
def forecast_frame() -> pd.DataFrame:
    ds = pd.date_range("2024-01-01", periods=6, freq="h")
    rows = []
    for model_id in ("naive", "mstl"):
        for i, ts in enumerate(ds):
            rows.append(
                {
                    "unique_id": "a",
                    "model_id": model_id,
                    "ds": ts,
                    "y": 10.0 + i,
                    "y_hat": 10.0 + i + (0.5 if model_id == "mstl" else -0.5),
                    "q_0250": 9.0 + i,
                    "q_9750": 11.0 + i,
                }
            )
    return pd.DataFrame(rows)


class TestPlotForecastOverlay:
    def test_dibuja_historico_real_y_una_linea_por_modelo(
        self, forecast_frame: pd.DataFrame
    ) -> None:
        context = pd.DataFrame(
            {"ds": pd.date_range("2023-12-31", periods=3, freq="h"), "y": [1.0, 2.0, 3.0]}
        )
        colors = plots.model_color_map(["naive", "mstl"])
        fig = plots.plot_forecast_overlay(context, forecast_frame, model_colors=colors)
        names = [trace.name for trace in fig.data]
        assert "Historico" in names
        assert "naive" in names
        assert "mstl" in names

    def test_sin_columnas_de_cuantil_no_dibuja_banda(self) -> None:
        context = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=1, freq="h"), "y": [1.0]})
        forecasts = pd.DataFrame(
            {
                "unique_id": ["a"],
                "model_id": ["naive"],
                "ds": pd.date_range("2024-01-01", periods=1, freq="h"),
                "y_hat": [1.0],
            }
        )
        fig = plots.plot_forecast_overlay(
            context, forecasts, model_colors=plots.model_color_map(["naive"])
        )
        # Solo la traza del historico y la del propio modelo: sin banda no hay
        # trazas invisibles de relleno.
        assert len(fig.data) == 2


class TestPlotResiduals:
    def test_una_traza_por_modelo(self, forecast_frame: pd.DataFrame) -> None:
        colors = plots.model_color_map(["naive", "mstl"])
        fig = plots.plot_residuals(forecast_frame, model_colors=colors)
        assert len(fig.data) == 2

    def test_el_residuo_es_real_menos_prediccion(self, forecast_frame: pd.DataFrame) -> None:
        colors = plots.model_color_map(["naive", "mstl"])
        fig = plots.plot_residuals(forecast_frame, model_colors=colors)
        naive_trace = next(t for t in fig.data if t.name == "naive")
        # Por construccion, naive tiene y_hat = y - 0.5, asi que el residuo es +0.5.
        np.testing.assert_allclose(np.asarray(naive_trace.y), 0.5)


# --------------------------------------------------------------------------- #
# 9. Leaderboard: precision vs coste, Diebold-Mariano
# --------------------------------------------------------------------------- #


class TestPlotAccuracyVsCost:
    def test_una_traza_por_modelo(self) -> None:
        leaderboard = pd.DataFrame(
            {"model_id": ["naive", "mstl"], "fit_seconds_total": [0.0, 4.2], "mase": [1.0, 0.7]}
        )
        fig = plots.plot_accuracy_vs_cost(
            leaderboard, model_colors=plots.model_color_map(["naive", "mstl"])
        )
        assert len(fig.data) == 2

    def test_coste_cero_no_revienta_la_escala_log(self) -> None:
        # `fit_seconds_total=0` es real (Chronos zero-shot); un eje log no
        # deberia lanzar por eso.
        leaderboard = pd.DataFrame(
            {"model_id": ["chronos"], "fit_seconds_total": [0.0], "mase": [1.0]}
        )
        fig = plots.plot_accuracy_vs_cost(
            leaderboard, model_colors=plots.model_color_map(["chronos"])
        )
        assert isinstance(fig, go.Figure)


class TestPlotDmHeatmap:
    def test_matriz_cuadrada_del_tamano_del_universo_de_modelos(self) -> None:
        dm_matrix = pd.DataFrame(
            {
                "model_a": ["naive", "mstl"],
                "model_b": ["mstl", "naive"],
                "stat": [-2.1, 2.1],
                "p_value": [0.03, 0.03],
                "n_obs": [500, 500],
                "mean_difference": [-0.2, 0.2],
                "degenerate": [False, False],
            }
        )
        fig = plots.plot_dm_heatmap(dm_matrix)
        heatmap = fig.data[0]
        assert heatmap.z.shape == (2, 2)
        assert list(heatmap.x) == ["mstl", "naive"]


# --------------------------------------------------------------------------- #
# 10. Anomalias
# --------------------------------------------------------------------------- #


class TestAnomalyThreshold:
    def test_alfa_pequeno_da_un_umbral_alto(self) -> None:
        assert plots.anomaly_threshold(0.01) > plots.anomaly_threshold(0.1)

    def test_es_menos_log10_de_alfa(self) -> None:
        assert plots.anomaly_threshold(0.05) == pytest.approx(-np.log10(0.05))

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_fuera_de_rango_lanza(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            plots.anomaly_threshold(alpha)


class TestPlotAnomalySeries:
    def test_marca_solo_los_puntos_que_superan_el_umbral(self) -> None:
        ds = pd.date_range("2024-01-01", periods=5, freq="h")
        series = pd.DataFrame({"ds": ds, "y": [1.0, 2.0, 30.0, 2.0, 1.0]})
        scores = pd.DataFrame(
            {"ds": ds, "score": [0.1, 0.1, 5.0, 0.1, 0.1], "scorable": [True] * 5}
        )
        truth = pd.DataFrame(columns=["event_id", "ds"])
        fig = plots.plot_anomaly_series(
            series, scores, truth, threshold=1.5, color=plots.CATEGORICAL[0], unique_id="a"
        )
        flagged = next(t for t in fig.data if t.name and t.name.startswith("Marcado"))
        assert len(flagged.x) == 1
        assert flagged.y[0] == 30.0

    def test_sin_ningun_punto_marcado_no_hay_traza_de_marcadores(self) -> None:
        ds = pd.date_range("2024-01-01", periods=3, freq="h")
        series = pd.DataFrame({"ds": ds, "y": [1.0, 1.0, 1.0]})
        scores = pd.DataFrame({"ds": ds, "score": [0.1, 0.1, 0.1], "scorable": [True] * 3})
        truth = pd.DataFrame(columns=["event_id", "ds"])
        fig = plots.plot_anomaly_series(
            series, scores, truth, threshold=5.0, color=plots.CATEGORICAL[0], unique_id="a"
        )
        assert not any(t.name and t.name.startswith("Marcado") for t in fig.data)


# --------------------------------------------------------------------------- #
# 11. Explicabilidad
# --------------------------------------------------------------------------- #


class TestPlotTftVariableAttention:
    def test_una_traza_por_bloque(self) -> None:
        table = pd.DataFrame(
            {
                "kind": ["attention_variable"] * 3,
                "feature": ["temp_c", "observed_target", "temp_c"],
                "block": ["future", "past", "past"],
                "value": [1.0, 0.62, 0.38],
            }
        )
        fig = plots.plot_tft_variable_attention(table)
        assert len(fig.data) == 2  # past, future


class TestPlotTftTemporalAttention:
    def test_ignora_las_filas_de_atencion_de_variable(self) -> None:
        table = pd.DataFrame(
            {
                "kind": ["attention_variable", "attention_temporal", "attention_temporal"],
                "offset": [np.nan, -1.0, 0.0],
                "value": [1.0, 0.4, 0.6],
            }
        )
        fig = plots.plot_tft_temporal_attention(table)
        assert len(fig.data[0].x) == 2


class TestPlotPredictionDecomposition:
    def test_la_cascada_tiene_cinco_barras(self) -> None:
        ds = pd.Timestamp("2024-01-01")
        components = pd.DataFrame(
            {
                "ds": [ds],
                "observed": [10.0],
                "trend": [8.0],
                "seasonal_24": [1.5],
                "seasonal_168": [0.3],
                "resid": [0.2],
            }
        )
        fig = plots.plot_prediction_decomposition(components, unique_id="a", ds=ds)
        assert len(fig.data[0].y) == 5

    def test_instante_ausente_lanza_keyerror(self) -> None:
        components = pd.DataFrame(
            {
                "ds": [pd.Timestamp("2024-01-01")],
                "observed": [10.0],
                "trend": [8.0],
                "seasonal_24": [1.5],
                "seasonal_168": [0.3],
                "resid": [0.2],
            }
        )
        with pytest.raises(KeyError):
            plots.plot_prediction_decomposition(
                components, unique_id="a", ds=pd.Timestamp("2099-01-01")
            )
