"""Agregacion en eventos y emparejamiento con la verdad.

El test que mas importa es el de la asignacion uno a uno: sin ella, partir un
evento real en trozos subiria el recall sin coste, que es exactamente como las
metricas ajustadas por punto inflan resultados.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chronolab.anomaly.events import (
    EVENT_COLUMNS,
    MATCH_COLUMNS,
    aggregate_events,
    match_events,
    truth_events,
)
from chronolab.types import DetectorId

DETECTOR = DetectorId("probe_detector")
ALPHA = 0.05
START = pd.Timestamp("2023-03-01")


def _scores(
    pattern: str,
    *,
    uid: str = "s00",
    severity: list[float] | None = None,
    side: list[int] | None = None,
    scorable: list[bool] | None = None,
) -> pd.DataFrame:
    """Trama de scores a partir de un patron ``#`` marcado, ``.`` no marcado."""
    n = len(pattern)
    marked = np.array([character == "#" for character in pattern])
    threshold = -math.log10(ALPHA)
    score = np.where(marked, threshold + 1.0, threshold - 1.0)
    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": pd.date_range(START, periods=n, freq="h"),
            "score": score.astype(np.float32),
            "scorable": np.ones(n, dtype=bool) if scorable is None else np.array(scorable),
            "severity": (
                np.where(marked, 0.5, -0.2) if severity is None else np.array(severity)
            ).astype(np.float32),
            "side": (np.where(marked, 1, 0) if side is None else np.array(side)).astype(np.int8),
        }
    )


def _detected(spans: list[tuple[int, int]], *, uid: str = "s00") -> pd.DataFrame:
    """Eventos detectados sinteticos, por posiciones inclusivas sobre la rejilla."""
    rows = []
    for start, end in spans:
        start_ds = START + pd.Timedelta(hours=start)
        rows.append(
            {
                "detector_id": str(DETECTOR),
                "unique_id": uid,
                "event_id": f"{DETECTOR}|{uid}|{ALPHA:.4f}|{start_ds:%Y%m%dT%H%M}",
                "alpha": ALPHA,
                "start_ds": start_ds,
                "end_ds": START + pd.Timedelta(hours=end),
                "n_points": end - start + 1,
                "duration_steps": end - start + 1,
                "peak_score": 2.0,
                "peak_severity": 0.5,
                "cum_severity": 0.5 * (end - start + 1),
                "peak_ds": start_ds,
                "direction": "over",
            }
        )
    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def _truth(spans: list[tuple[int, int]], *, uid: str = "s00") -> pd.DataFrame:
    """Verdad puntual a partir de tramos inclusivos."""
    rows = []
    for index, (start, end) in enumerate(spans):
        for offset in range(start, end + 1):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": START + pd.Timedelta(hours=offset),
                    "is_anomaly": True,
                    "event_id": f"truth_{index}",
                    "anomaly_type": "level_shift",
                }
            )
    return pd.DataFrame(rows)


class TestAgregacion:
    def test_una_tirada_contigua_es_un_evento(self) -> None:
        events = aggregate_events(_scores("..###..."), detector_id=DETECTOR, alpha=ALPHA)
        assert len(events) == 1
        assert events.loc[0, "n_points"] == 3
        assert events.loc[0, "duration_steps"] == 3
        assert events.loc[0, "start_ds"] == START + pd.Timedelta(hours=2)
        assert events.loc[0, "end_ds"] == START + pd.Timedelta(hours=4)

    def test_un_hueco_corto_no_parte_el_evento(self) -> None:
        # Una anomalia real casi nunca produce una tirada ininterrumpida; sin
        # tolerancia, un solo punto de vuelta dentro de la banda destroza la
        # precision a nivel de evento.
        events = aggregate_events(_scores("##.##"), detector_id=DETECTOR, alpha=ALPHA, merge_gap=1)
        assert len(events) == 1
        assert events.loc[0, "n_points"] == 4
        assert events.loc[0, "duration_steps"] == 5

    def test_un_hueco_largo_si_lo_parte(self) -> None:
        events = aggregate_events(
            _scores("##...##"), detector_id=DETECTOR, alpha=ALPHA, merge_gap=1
        )
        assert len(events) == 2

    def test_la_tolerancia_declarada_gobierna_la_fusion(self) -> None:
        pattern = "##..##"
        assert (
            len(aggregate_events(_scores(pattern), detector_id=DETECTOR, alpha=ALPHA, merge_gap=2))
            == 1
        )
        assert (
            len(aggregate_events(_scores(pattern), detector_id=DETECTOR, alpha=ALPHA, merge_gap=1))
            == 2
        )

    def test_min_points_no_filtra_picos_por_defecto(self) -> None:
        # Con alfa pequeno el pico de un solo punto es uno de los cinco tipos que
        # se inyectan: filtrarlo por defecto seria filtrar el objetivo.
        assert len(aggregate_events(_scores("..#.."), detector_id=DETECTOR, alpha=ALPHA)) == 1
        filtered = aggregate_events(
            _scores("..#.."), detector_id=DETECTOR, alpha=ALPHA, min_points=2
        )
        assert filtered.empty

    def test_las_severidades_se_agregan_como_maximo_y_como_suma(self) -> None:
        # peak_severity sola no distingue una hora muy fuera de seis horas
        # ligeramente fuera, que son incidentes distintos.
        scores = _scores("..###..", severity=[-0.1, -0.1, 0.2, 1.5, 0.4, -0.1, -0.1])
        events = aggregate_events(scores, detector_id=DETECTOR, alpha=ALPHA)
        assert events.loc[0, "peak_severity"] == pytest.approx(1.5, rel=1e-6)
        assert events.loc[0, "cum_severity"] == pytest.approx(2.1, rel=1e-5)
        assert events.loc[0, "peak_ds"] == START + pd.Timedelta(hours=3)

    def test_el_sentido_separa_un_pico_de_una_caida(self) -> None:
        over = aggregate_events(_scores("###", side=[1, 1, 1]), detector_id=DETECTOR, alpha=ALPHA)
        under = aggregate_events(
            _scores("###", side=[-1, -1, -1]), detector_id=DETECTOR, alpha=ALPHA
        )
        mixed = aggregate_events(_scores("###", side=[1, -1, 1]), detector_id=DETECTOR, alpha=ALPHA)
        assert over.loc[0, "direction"] == "over"
        assert under.loc[0, "direction"] == "under"
        assert mixed.loc[0, "direction"] == "mixed"

    def test_un_punto_no_puntuable_no_se_marca(self) -> None:
        scores = _scores("###", scorable=[True, False, True])
        events = aggregate_events(scores, detector_id=DETECTOR, alpha=ALPHA, merge_gap=1)
        assert len(events) == 1
        assert events.loc[0, "n_points"] == 2
        assert events.loc[0, "duration_steps"] == 3

    def test_el_identificador_de_evento_depende_de_alfa(self) -> None:
        # Los eventos son funcion de alfa: sin ella en la clave, dos filas de la
        # tabla con distinto alfa colisionarian.
        scores = _scores(".###.")
        loose = aggregate_events(scores, detector_id=DETECTOR, alpha=0.1)
        tight = aggregate_events(scores, detector_id=DETECTOR, alpha=0.05)
        assert loose.loc[0, "event_id"] != tight.loc[0, "event_id"]

    def test_sin_marcas_devuelve_la_tabla_vacia_con_su_esquema(self) -> None:
        events = aggregate_events(_scores("....."), detector_id=DETECTOR, alpha=ALPHA)
        assert events.empty
        assert list(events.columns) == list(EVENT_COLUMNS)

    @pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"merge_gap": -1}, {"min_points": 0}])
    def test_los_parametros_incoherentes_fallan(self, kwargs: dict[str, float]) -> None:
        defaults: dict[str, object] = {"detector_id": DETECTOR, "alpha": ALPHA}
        with pytest.raises(ValueError):
            aggregate_events(_scores("###"), **{**defaults, **kwargs})  # type: ignore[arg-type]


class TestVerdad:
    def test_colapsa_los_puntos_en_eventos(self) -> None:
        events = truth_events(_truth([(2, 5), (10, 11)]))
        assert len(events) == 2
        assert events.loc[0, "n_points"] == 4
        assert events.loc[0, "start_ds"] == START + pd.Timedelta(hours=2)
        assert events.loc[1, "end_ds"] == START + pd.Timedelta(hours=11)

    def test_una_verdad_sin_anomalias_da_la_tabla_vacia(self) -> None:
        truth = _truth([(1, 2)])
        truth["is_anomaly"] = False
        assert truth_events(truth).empty


class TestEmparejamiento:
    def test_un_evento_real_partido_en_tres_no_da_tres_aciertos(self) -> None:
        # Es la propiedad que impide reproducir el vicio del point-adjusted F1.
        detected = _detected([(0, 1), (4, 5), (8, 9)])
        matched = match_events(detected, truth_events(_truth([(0, 9)])), alpha=ALPHA)
        kinds = matched["match_kind"].value_counts()
        assert kinds.get("hit", 0) == 1
        assert kinds.get("false_alarm", 0) == 2
        assert kinds.get("missed", 0) == 0

    def test_el_acierto_se_lleva_el_de_mayor_solape(self) -> None:
        detected = _detected([(0, 1), (3, 9)])
        matched = match_events(detected, truth_events(_truth([(3, 9)])), alpha=ALPHA)
        hit = matched.loc[matched["match_kind"] == "hit"].iloc[0]
        assert hit["start_ds"] == START + pd.Timedelta(hours=3)

    def test_un_evento_real_sin_pareja_aparece_como_perdido(self) -> None:
        matched = match_events(_detected([(0, 1)]), truth_events(_truth([(50, 55)])), alpha=ALPHA)
        missed = matched.loc[matched["match_kind"] == "missed"]
        assert len(missed) == 1
        assert missed.iloc[0]["matched_truth_event_id"] == "truth_0"
        assert missed.iloc[0]["event_id"] is None
        assert missed.iloc[0]["alpha"] == ALPHA

    def test_sin_nada_detectado_la_verdad_sigue_contandose(self) -> None:
        empty = _detected([])
        matched = match_events(empty, truth_events(_truth([(1, 3)])), alpha=ALPHA)
        assert list(matched["match_kind"]) == ["missed"]

    def test_la_tolerancia_convierte_un_fallo_por_poco_en_acierto(self) -> None:
        detected = _detected([(10, 12)])
        truth = truth_events(_truth([(13, 15)]))
        assert match_events(detected, truth, alpha=ALPHA)["match_kind"].tolist() == [
            "false_alarm",
            "missed",
        ]
        with_tol = match_events(detected, truth, alpha=ALPHA, tol=pd.Timedelta(hours=1))
        assert with_tol["match_kind"].tolist() == ["hit"]

    def test_las_series_no_se_mezclan(self) -> None:
        detected = _detected([(0, 5)], uid="s00")
        truth = truth_events(_truth([(0, 5)], uid="s01"))
        kinds = match_events(detected, truth, alpha=ALPHA)["match_kind"].tolist()
        assert sorted(kinds) == ["false_alarm", "missed"]

    def test_mezclar_alfas_es_un_error(self) -> None:
        detected = _detected([(0, 1)])
        detected.loc[0, "alpha"] = 0.1
        with pytest.raises(ValueError, match="alpha"):
            match_events(detected, truth_events(_truth([(0, 1)])), alpha=ALPHA)

    def test_conserva_el_esquema_de_la_tabla(self) -> None:
        matched = match_events(_detected([(0, 1)]), truth_events(_truth([(0, 1)])), alpha=ALPHA)
        assert list(matched.columns) == list(MATCH_COLUMNS)
