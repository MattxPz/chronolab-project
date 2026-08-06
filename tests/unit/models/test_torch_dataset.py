"""Escalado por serie y ventanas deslizantes: valores exactos y cortes temporales.

Este fichero **no** esta marcado `slow` ni lleva `importorskip`: es la mitad de
`chronolab.models.torch` que a proposito no depende de `torch`, y por tanto la
que se puede ejercitar en el entorno por defecto de CI. Lo que se comprueba
aqui es la aritmetica de indices y de estadisticos, que es justo donde vive el
riesgo de fuga; el tensor se prueba aparte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronolab.models.torch.dataset import (
    SeriesScaler,
    build_windows,
    context_matrix,
)
from chronolab.panel import Panel, PanelSpec
from chronolab.types import DatasetId

SPEC = PanelSpec(
    dataset_id=DatasetId("mini"),
    freq="h",
    seasonalities=(2, 4),
    futr_exog=("temp_c",),
    hist_exog=("voltage",),
)


def _panel(values: dict[str, list[float]], *, temp: list[float] | None = None) -> Panel:
    """Panel de juguete con los valores exactos que se le pasen."""
    parts = []
    for uid, series in values.items():
        n = len(series)
        parts.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range("2023-01-02", periods=n, freq="h"),
                    "y": series,
                    "temp_c": temp if temp is not None else np.arange(n, dtype=float),
                    "voltage": np.full(n, 230.0),
                }
            )
        )
    return Panel(df=pd.concat(parts, ignore_index=True), spec=SPEC)


class TestSeriesScaler:
    def test_estandariza_cada_serie_con_sus_propios_estadisticos(self) -> None:
        panel = _panel({"a": [1.0, 2.0, 3.0], "b": [100.0, 200.0, 300.0]})
        scaler = SeriesScaler.fit(panel)

        assert scaler.target_mean["a"] == pytest.approx(2.0)
        assert scaler.target_mean["b"] == pytest.approx(200.0)
        # Series de escalas muy distintas quedan ambas centradas en cero.
        assert scaler.transform_target(np.array([2.0]), "a")[0] == pytest.approx(0.0)
        assert scaler.transform_target(np.array([200.0]), "b")[0] == pytest.approx(0.0)

    def test_inverse_deshace_transform(self) -> None:
        panel = _panel({"a": [1.0, 5.0, 9.0, 13.0]})
        scaler = SeriesScaler.fit(panel)
        original = np.array([1.0, 7.0, 13.0])

        recovered = scaler.inverse_target(scaler.transform_target(original, "a"), "a")

        np.testing.assert_allclose(recovered, original, atol=1e-9)

    def test_una_serie_constante_no_produce_infinitos(self) -> None:
        panel = _panel({"a": [5.0, 5.0, 5.0, 5.0]})
        scaler = SeriesScaler.fit(panel)
        scaled = scaler.transform_target(np.array([5.0, 5.0]), "a")
        assert np.isfinite(scaled).all()

    def test_solo_ve_el_tramo_que_se_le_pasa(self) -> None:
        # La barrera L2 es que `fit` recibe el train ya recortado; aqui se
        # comprueba lo que se sigue de eso: los estadisticos de un prefijo no
        # dependen de lo que venga despues.
        completo = _panel({"a": [1.0, 2.0, 3.0, 4.0, 1000.0]})
        prefijo = completo.slice(completo.first_ds, completo.first_ds + pd.Timedelta(hours=3))

        assert SeriesScaler.fit(prefijo).target_mean["a"] == pytest.approx(2.5)
        assert SeriesScaler.fit(completo).target_mean["a"] != pytest.approx(2.5)

    def test_las_exogenas_se_escalan_en_el_orden_declarado(self) -> None:
        panel = _panel({"a": [1.0, 2.0, 3.0, 4.0]})
        scaler = SeriesScaler.fit(panel, ("temp_c",))
        matrix = scaler.transform_exog(panel.df)

        assert matrix.shape == (4, 1)
        assert matrix.mean() == pytest.approx(0.0, abs=1e-6)

    def test_sin_exogenas_la_matriz_tiene_cero_columnas(self) -> None:
        panel = _panel({"a": [1.0, 2.0, 3.0]})
        assert SeriesScaler.fit(panel).transform_exog(panel.df).shape == (3, 0)

    def test_los_nan_no_contaminan_los_estadisticos(self) -> None:
        panel = _panel({"a": [1.0, np.nan, 3.0]})
        scaler = SeriesScaler.fit(panel)
        assert scaler.target_mean["a"] == pytest.approx(2.0)


class TestBuildWindows:
    def test_el_contexto_termina_justo_antes_del_horizonte(self) -> None:
        # y = 0..9; con input_size=3 y h=2 la primera ventana tiene contexto
        # [0,1,2] y objetivo [3,4]: el 3 no puede estar en la entrada.
        panel = _panel({"a": list(np.arange(10, dtype=float))})
        scaler = SeriesScaler.fit(panel)

        batch = build_windows(panel, scaler, input_size=3, h=2)

        primero_contexto = scaler.inverse_target(batch.context[0, :, 0], "a")
        primero_objetivo = scaler.inverse_target(batch.target[0], "a")
        np.testing.assert_allclose(primero_contexto, [0.0, 1.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(primero_objetivo, [3.0, 4.0], atol=1e-6)

    def test_numero_de_ventanas(self) -> None:
        panel = _panel({"a": list(np.arange(10, dtype=float))})
        scaler = SeriesScaler.fit(panel)
        # Posiciones de fin de contexto validas: 2..7 -> 6 ventanas.
        assert len(build_windows(panel, scaler, input_size=3, h=2)) == 6

    def test_las_ventanas_no_cruzan_la_frontera_entre_series(self) -> None:
        panel = _panel({"a": [0.0, 1.0, 2.0, 3.0], "b": [100.0, 101.0, 102.0, 103.0]})
        scaler = SeriesScaler.fit(panel)

        batch = build_windows(panel, scaler, input_size=2, h=1)

        # Cada ventana lleva su serie, y ninguna mezcla valores de las dos.
        assert set(batch.series) == {"a", "b"}
        for position, uid in enumerate(batch.series):
            context = scaler.inverse_target(batch.context[position, :, 0], uid)
            assert (context < 50).all() if uid == "a" else (context > 50).all()

    def test_una_ventana_con_hueco_en_el_objetivo_se_descarta(self) -> None:
        # Imputar la entrada es una decision de modelado; inventarse la
        # respuesta que se evalua no lo es.
        panel = _panel({"a": [0.0, 1.0, 2.0, np.nan, 4.0, 5.0]})
        scaler = SeriesScaler.fit(panel)

        batch = build_windows(panel, scaler, input_size=2, h=1)

        objetivos = scaler.inverse_target(batch.target[:, 0], "a")
        assert not np.isnan(objetivos).any()
        assert 3.0 not in set(np.round(objetivos, 6))

    def test_las_exogenas_futuras_cubren_el_tramo_a_predecir(self) -> None:
        panel = _panel({"a": list(np.arange(10, dtype=float))}, temp=list(np.arange(10, 20.0)))
        scaler = SeriesScaler.fit(panel, ("temp_c",))

        batch = build_windows(panel, scaler, input_size=3, h=2)

        assert batch.context.shape[2] == 2  # objetivo + temp_c
        assert batch.futr.shape == (len(batch), 2, 1)
        # La primera ventana predice las posiciones 3 y 4: temp 13 y 14.
        futura = batch.futr[0, :, 0] * scaler.exog_std["temp_c"] + scaler.exog_mean["temp_c"]
        np.testing.assert_allclose(futura, [13.0, 14.0], atol=1e-4)

    def test_un_tramo_demasiado_corto_no_produce_ventanas(self) -> None:
        panel = _panel({"a": [1.0, 2.0]})
        scaler = SeriesScaler.fit(panel)
        assert len(build_windows(panel, scaler, input_size=5, h=3)) == 0

    @pytest.mark.parametrize(("input_size", "h"), [(0, 2), (3, 0)])
    def test_rechaza_dimensiones_no_positivas(self, input_size: int, h: int) -> None:
        panel = _panel({"a": [1.0, 2.0, 3.0]})
        scaler = SeriesScaler.fit(panel)
        with pytest.raises(ValueError, match=">= 1"):
            build_windows(panel, scaler, input_size=input_size, h=h)

    def test_estabilidad_por_prefijos(self) -> None:
        # Version local del test T1 de docs/ARCHITECTURE.md: las ventanas
        # calculadas sobre un prefijo son exactamente las primeras de las
        # calculadas sobre el panel entero. Si alguna operacion mirase hacia
        # delante, esta igualdad se rompe.
        valores = list(np.arange(20, dtype=float))
        completo = _panel({"a": valores})
        corte = completo.first_ds + pd.Timedelta(hours=11)
        prefijo = completo.slice(completo.first_ds, corte)

        scaler = SeriesScaler.fit(prefijo)  # mismo escalador en ambos lados
        de_prefijo = build_windows(prefijo, scaler, input_size=3, h=2)
        de_completo = build_windows(completo, scaler, input_size=3, h=2)

        n = len(de_prefijo)
        np.testing.assert_allclose(de_prefijo.context, de_completo.context[:n], atol=1e-6)
        np.testing.assert_allclose(de_prefijo.target, de_completo.target[:n], atol=1e-6)


class TestContextMatrix:
    def test_toma_el_ultimo_contexto_de_cada_serie(self) -> None:
        panel = _panel({"a": list(np.arange(6, dtype=float))})
        scaler = SeriesScaler.fit(panel)

        matrix, ids = context_matrix(panel, scaler, input_size=3)

        assert ids == ("a",)
        np.testing.assert_allclose(
            scaler.inverse_target(matrix[0, :, 0], "a"), [3.0, 4.0, 5.0], atol=1e-6
        )

    def test_una_serie_mas_corta_que_el_contexto_falla_claro(self) -> None:
        panel = _panel({"a": [1.0, 2.0]})
        scaler = SeriesScaler.fit(panel)
        with pytest.raises(ValueError, match="el contexto exige"):
            context_matrix(panel, scaler, input_size=5)
