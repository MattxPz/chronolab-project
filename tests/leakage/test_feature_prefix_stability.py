"""T1: estabilidad por prefijos de las features (docs/ARCHITECTURE.md §8.1).

Para todo corte `t`, ``features(panel[:t])`` tiene que coincidir exactamente con
``features(panel)[:t]``. La gracia de la propiedad es que es **general**: no hay
que enumerar las operaciones prospectivas para detectarlas, porque cualquier
operacion que mire hacia delante rompe la igualdad, sea cual sea su forma. Un
lag negativo, una media centrada, un `bfill`, un `interpolate` en ambos sentidos
o un escalador ajustado con toda la serie: todos fallan aqui.

El ultimo test es el control positivo del propio T1. Construye a mano una media
movil **centrada** —que `features.ops` no ofrece— y comprueba que la propiedad se
rompe. Sin el, un T1 que pasara siempre por un error de montaje seria
indistinguible de un T1 que pasa porque el codigo es correcto.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chronolab.features import ops
from chronolab.panel import Panel
from tests.fixtures.synthetic import DAILY, WEEKLY, hourly_spec, make_hourly_frame


@pytest.fixture(scope="module")
def panel() -> Panel:
    return Panel(df=make_hourly_frame(n_series=2, n_hours=WEEKLY * 4, seed=5), spec=hourly_spec())


def _features(panel: Panel) -> pd.DataFrame:
    """Un conjunto de features con todas las primitivas retrospectivas del modulo."""
    features = [
        ops.lag(panel, "y", DAILY),
        ops.diff(panel, "y", 1),
        ops.roll(panel, "y", DAILY, shift=1, stat="mean"),
        ops.roll(panel, "y", WEEKLY, shift=DAILY, stat="std"),
        ops.roll(panel, "voltage", 6, shift=2, stat="max"),
        ops.expand(panel, "y", shift=1, stat="mean"),
        ops.ewm(panel, "y", halflife=12.0, shift=1),
        ops.lag(panel, ops.roll(panel, "y", DAILY, shift=1), DAILY),  # composicion
    ]
    frame = panel.df[["unique_id", "ds"]].copy()
    for feature in features:
        frame[feature.name] = feature.values
    return frame.reset_index(drop=True)


@pytest.mark.parametrize("horas", [DAILY, WEEKLY, WEEKLY * 2, WEEKLY * 3 + 5])
def test_las_features_de_un_prefijo_coinciden_con_el_prefijo_de_las_features(
    panel: Panel, horas: int
) -> None:
    corte = panel.first_ds + pd.Timedelta(hours=horas)

    sobre_el_prefijo = _features(panel.slice(panel.first_ds, corte))
    prefijo_del_total = _features(panel)
    prefijo_del_total = prefijo_del_total[prefijo_del_total["ds"] <= corte].reset_index(drop=True)

    pd.testing.assert_frame_equal(sobre_el_prefijo, prefijo_del_total)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(horas=st.integers(min_value=DAILY, max_value=WEEKLY * 4 - 1))
def test_la_propiedad_se_cumple_en_cualquier_corte(panel: Panel, horas: int) -> None:
    corte = panel.first_ds + pd.Timedelta(hours=horas)

    sobre_el_prefijo = _features(panel.slice(panel.first_ds, corte))
    prefijo_del_total = _features(panel)
    prefijo_del_total = prefijo_del_total[prefijo_del_total["ds"] <= corte].reset_index(drop=True)

    pd.testing.assert_frame_equal(sobre_el_prefijo, prefijo_del_total)


def test_las_ventanas_moviles_no_admiten_desplazamiento_cero(panel: Panel) -> None:
    # Una media movil que incluye el instante `t` usa el valor que se quiere
    # predecir. La firma no lo permite y no hay parametro `center`.
    for shift in (0, -1):
        with pytest.raises(ValueError, match="ventana movil debe ser >= 1"):
            ops.roll(panel, "y", DAILY, shift=shift)
        with pytest.raises(ValueError, match="ventana movil debe ser >= 1"):
            ops.expand(panel, "y", shift=shift)
        with pytest.raises(ValueError, match="ventana movil debe ser >= 1"):
            ops.ewm(panel, "y", halflife=6.0, shift=shift)

    # Y `center` no es un argumento que se pueda pasar de ninguna forma.
    with pytest.raises(TypeError):
        ops.roll(panel, "y", DAILY, center=True)  # type: ignore[call-arg]


def test_lead_sobre_una_exogena_futura_es_prospectivo_a_proposito(panel: Panel) -> None:
    """`lead` queda fuera de T1 porque su definicion es mirar hacia delante.

    Es legitimo unicamente sobre columnas con `max_lead` infinito —el calendario,
    una prevision— cuyo valor futuro si se conoce en el cutoff. Justamente por
    eso no cumple la estabilidad por prefijos: el prefijo no contiene ese futuro,
    pero en un backtest lo entrega el `FutrProvider`, no el panel.
    """
    corte = panel.first_ds + pd.Timedelta(hours=WEEKLY)
    adelantada = ops.lead(panel, "temp_c", 3)
    prefijo = ops.lead(panel.slice(panel.first_ds, corte), "temp_c", 3)

    completa = adelantada.values[panel.df["ds"].to_numpy() <= corte.to_numpy()]
    assert prefijo.values.tail(3).isna().all()
    assert not pd.Series(completa).tail(3).isna().all()


def test_una_media_centrada_rompe_la_propiedad(panel: Panel) -> None:
    """Control positivo de T1: si se introduce la fuga, el test la ve.

    La media centrada no existe en `features.ops`, asi que hay que fabricarla a
    mano con pandas. Ese es justamente el punto: para escribirla hay que salir
    del modulo.
    """

    def centrada(panel: Panel) -> pd.DataFrame:
        frame = panel.df[["unique_id", "ds"]].copy()
        frame["fuga"] = (
            panel.df.groupby("unique_id", sort=False)["y"]
            .rolling(DAILY, center=True, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        return frame.reset_index(drop=True)

    corte = panel.first_ds + pd.Timedelta(hours=WEEKLY)
    sobre_el_prefijo = centrada(panel.slice(panel.first_ds, corte))
    prefijo_del_total = centrada(panel)
    prefijo_del_total = prefijo_del_total[prefijo_del_total["ds"] <= corte].reset_index(drop=True)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(sobre_el_prefijo, prefijo_del_total)
