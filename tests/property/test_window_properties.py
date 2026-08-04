"""Propiedades de `Window` verificadas con hypothesis.

Una ventana bien formada cumple las mismas relaciones aritmeticas sea cual sea su
tamano. Comprobarlas por muestreo detecta los off-by-one que un ejemplo concreto
elegido a mano no toca.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from chronolab.evaluation.splitters import Window

BASE = pd.Timestamp("2023-01-02")


def _build(train_len: int, gap: int, h: int, window_id: int) -> Window:
    cutoff = BASE + pd.Timedelta(hours=train_len - 1)
    first_pred = cutoff + pd.Timedelta(hours=gap + 1)
    return Window(
        window_id=window_id,
        stage="dev",
        train_start=BASE,
        cutoff=cutoff,
        first_pred=first_pred,
        last_pred=first_pred + pd.Timedelta(hours=h - 1),
        h=h,
        gap=gap,
    )


@given(
    train_len=st.integers(min_value=1, max_value=10_000),
    gap=st.integers(min_value=0, max_value=168),
    h=st.integers(min_value=1, max_value=336),
    window_id=st.integers(min_value=0, max_value=1_000),
)
def test_la_evaluacion_empieza_siempre_despues_del_cutoff(
    train_len: int, gap: int, h: int, window_id: int
) -> None:
    window = _build(train_len, gap, h, window_id)
    assert window.first_pred > window.cutoff


@given(
    train_len=st.integers(min_value=1, max_value=10_000),
    gap=st.integers(min_value=0, max_value=168),
    h=st.integers(min_value=1, max_value=336),
)
def test_el_gap_separa_exactamente_lo_declarado(train_len: int, gap: int, h: int) -> None:
    window = _build(train_len, gap, h, 0)
    assert window.first_pred - window.cutoff == pd.Timedelta(hours=gap + 1)


@given(
    train_len=st.integers(min_value=1, max_value=10_000),
    gap=st.integers(min_value=0, max_value=168),
    h=st.integers(min_value=1, max_value=336),
)
def test_el_tramo_evaluado_tiene_h_pasos(train_len: int, gap: int, h: int) -> None:
    window = _build(train_len, gap, h, 0)
    assert window.last_pred - window.first_pred == pd.Timedelta(hours=h - 1)


@given(
    gap=st.integers(min_value=0, max_value=168),
    h=st.integers(min_value=1, max_value=336),
)
def test_el_adelanto_crece_un_paso_por_h_step(gap: int, h: int) -> None:
    window = _build(1_000, gap, h, 0)
    leads = [window.lead(step) for step in range(1, h + 1)]
    assert leads == list(range(gap + 1, gap + h + 1))


@given(
    train_len=st.integers(min_value=1, max_value=10_000),
    gap=st.integers(min_value=0, max_value=168),
    h=st.integers(min_value=1, max_value=336),
)
def test_el_entrenamiento_nunca_solapa_con_la_evaluacion(train_len: int, gap: int, h: int) -> None:
    window = _build(train_len, gap, h, 0)
    assert window.train_start <= window.cutoff < window.first_pred <= window.last_pred
