"""Features de calendario y festivos: leidas en hora local, alineadas al indice UTC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronolab.data.calendar import calendar_features, fourier_terms, holiday_flags


class TestHolidayFlags:
    def test_ano_nuevo_es_festivo_en_espana(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-01-01 10:00"]))
        result = holiday_flags(ds, country="ES", tz_display="Europe/Madrid")
        assert result.iloc[0]

    def test_reyes_es_festivo_en_espana(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-01-06 10:00"]))
        result = holiday_flags(ds, country="ES", tz_display="Europe/Madrid")
        assert result.iloc[0]

    def test_un_dia_laborable_cualquiera_no_es_festivo(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-06-15 10:00"]))
        result = holiday_flags(ds, country="ES", tz_display="Europe/Madrid")
        assert not result.iloc[0]

    def test_un_festivo_cerca_de_medianoche_depende_del_huso_de_lectura(self) -> None:
        # 2024-01-01 00:30 UTC es 2024-01-01 01:30 en Madrid (festivo) pero
        # 2023-12-31 19:30 en Nueva York (no festivo): la fecha civil, y por
        # tanto el festivo, depende del huso de lectura.
        ds = pd.Series(pd.to_datetime(["2024-01-01 00:30"]))
        en_madrid = holiday_flags(ds, country="ES", tz_display="Europe/Madrid")
        en_nueva_york = holiday_flags(ds, country="US", tz_display="America/New_York")
        assert en_madrid.iloc[0]
        assert not en_nueva_york.iloc[0]

    def test_conserva_el_indice_de_entrada(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-01-01"]), index=[42])
        result = holiday_flags(ds, country="ES")
        assert list(result.index) == [42]

    def test_rechaza_entrada_tz_aware(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-01-01"])).dt.tz_localize("UTC")
        with pytest.raises(ValueError, match="UTC ingenuo"):
            holiday_flags(ds, country="ES")


class TestCalendarFeatures:
    def test_columnas_esperadas_sin_pais(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-01-01 12:00"]))
        result = calendar_features(ds)
        expected = {
            "ds",
            "hour",
            "dayofweek",
            "month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        }
        assert expected.issubset(result.columns)
        assert "is_holiday" not in result.columns

    def test_anade_is_holiday_solo_si_se_pide_pais(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-01-01 12:00"]))
        result = calendar_features(ds, country="ES", tz_display="Europe/Madrid")
        assert "is_holiday" in result.columns
        assert result["is_holiday"].iloc[0]

    def test_hora_local_difiere_de_la_hora_utc_por_el_huso(self) -> None:
        # 23:30 UTC de un dia es ya 00:30 del dia siguiente en Madrid en verano.
        ds = pd.Series(pd.to_datetime(["2024-07-01 23:30"]))
        result = calendar_features(ds, tz_display="Europe/Madrid")
        assert result["hour"].iloc[0] == 1
        assert result["dayofweek"].iloc[0] == 1  # martes 2 de julio en Madrid

    def test_fin_de_semana_marca_sabado_y_domingo(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-06-15 10:00", "2024-06-16 10:00", "2024-06-17 10:00"]))
        result = calendar_features(ds)
        assert result["is_weekend"].tolist() == [True, True, False]

    def test_hour_sin_cos_son_periodicos_en_24(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"]))
        result = calendar_features(ds)
        assert result["hour_sin"].iloc[0] == pytest.approx(result["hour_sin"].iloc[1], abs=1e-6)
        assert result["hour_cos"].iloc[0] == pytest.approx(result["hour_cos"].iloc[1], abs=1e-6)

    def test_medianoche_tiene_seno_nulo_y_coseno_maximo(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-01-01 00:00"]))
        result = calendar_features(ds)
        assert result["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_conserva_el_indice_de_entrada(self) -> None:
        ds = pd.Series(pd.to_datetime(["2024-01-01"]), index=[7])
        result = calendar_features(ds)
        assert list(result.index) == [7]


class TestFourierTerms:
    def test_columnas_por_periodo_y_armonico(self) -> None:
        ds = pd.Series(pd.date_range("2024-01-01", periods=48, freq="h"))
        result = fourier_terms(ds, periods={"daily": 24.0, "weekly": 168.0}, order=2)
        expected = {
            "fourier_daily_sin_1",
            "fourier_daily_cos_1",
            "fourier_daily_sin_2",
            "fourier_daily_cos_2",
            "fourier_weekly_sin_1",
            "fourier_weekly_cos_1",
            "fourier_weekly_sin_2",
            "fourier_weekly_cos_2",
        }
        assert expected.issubset(result.columns)

    def test_periodo_diario_se_repite_cada_24_pasos(self) -> None:
        ds = pd.Series(pd.date_range("2024-01-01", periods=48, freq="h"))
        result = fourier_terms(ds, periods={"daily": 24.0}, order=1)
        first_day = result["fourier_daily_sin_1"].iloc[:24].to_numpy()
        second_day = result["fourier_daily_sin_1"].iloc[24:].to_numpy()
        np.testing.assert_allclose(first_day, second_day, atol=1e-5)

    def test_sigue_siendo_correcto_con_huecos_en_la_rejilla(self) -> None:
        # A diferencia de una version basada en la posicion de la fila, el
        # calculo usa el tiempo transcurrido real: quitar una fila del medio
        # no debe desplazar el resto de la fase.
        full = pd.Series(pd.date_range("2024-01-01", periods=48, freq="h"))
        with_gap = full.drop(index=10).reset_index(drop=True)

        full_terms = fourier_terms(full, periods={"daily": 24.0}, order=1)
        gapped_terms = fourier_terms(with_gap, periods={"daily": 24.0}, order=1)

        # La fila 30 en la version completa es la misma marca de tiempo que la
        # fila 29 en la version con hueco (se elimino una fila anterior).
        assert full_terms["ds"].iloc[30] == gapped_terms["ds"].iloc[29]
        assert full_terms["fourier_daily_sin_1"].iloc[30] == pytest.approx(
            gapped_terms["fourier_daily_sin_1"].iloc[29], abs=1e-6
        )
