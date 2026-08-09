"""`chronolab.api.service`: la API sirve exactamente lo que hay en `live_dir`.

No corre ningun backtest ni ningun detector: los artefactos de estos tests se
construyen a mano, con el esquema exacto que `scripts/refresh_data.py`
escribe, para poder probar el filtrado y la serializacion sin `--extra ml` ni
red. Es el mismo motivo por el que `chronolab.api.service` no importa
`chronolab.models` ni `chronolab.anomaly` (docs/ARCHITECTURE.md §2.1): esta
suite corre en el job `quality` de CI, que no instala esos extras.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from chronolab.api.service import _get_live_dir, app

MODEL_A = "MSTL"
MODEL_B = "AutoETS"


def _write_manifest(live_dir: Path, *, generated_at: datetime, alpha: float = 0.05) -> None:
    payload = {
        "run_id": "01J9ZQK8N3F5XW1R7T2Y6H4B0C",
        "generated_at": generated_at.isoformat(),
        "alpha": alpha,
        "horizon": 6,
    }
    (live_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_forecasts(live_dir: Path) -> None:
    cutoff_a = pd.Timestamp("2024-08-07T00:00:00")
    cutoff_b = pd.Timestamp("2024-08-07T00:00:00")
    rows = []
    for h_step in range(1, 4):
        rows.append(
            {
                "unique_id": "ES",
                "model_id": MODEL_A,
                "window_id": 0,
                "stage": "holdout",
                "cutoff": cutoff_a,
                "ds": cutoff_a + pd.Timedelta(hours=h_step),
                "h_step": h_step,
                "y": 100.0 + h_step,
                "y_hat": 100.0 + h_step,
                "q_0250": 95.0 + h_step,
                "q_5000": 100.0 + h_step,
                "q_9750": 105.0 + h_step,
            }
        )
        rows.append(
            {
                "unique_id": "ES",
                "model_id": MODEL_B,
                "window_id": 0,
                "stage": "holdout",
                "cutoff": cutoff_b,
                "ds": cutoff_b + pd.Timedelta(hours=h_step),
                "h_step": h_step,
                "y": 100.0 + h_step,
                "y_hat": 101.0 + h_step,
                "q_0250": 96.0 + h_step,
                "q_5000": 101.0 + h_step,
                "q_9750": 106.0 + h_step,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(live_dir / "forecasts.parquet", index=False)


def _write_windows(live_dir: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "window_id": 0,
                "stage": "holdout",
                "train_start": pd.Timestamp("2024-07-01T00:00:00"),
                "cutoff": pd.Timestamp("2024-08-07T00:00:00"),
                "first_pred": pd.Timestamp("2024-08-07T01:00:00"),
                "last_pred": pd.Timestamp("2024-08-07T06:00:00"),
            }
        ]
    )
    frame.to_parquet(live_dir / "windows.parquet", index=False)


def _write_anomaly_scores(live_dir: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "unique_id": "ES",
                "ds": pd.Timestamp("2024-08-07T01:00:00"),
                "detector_id": "conformal_MSTL",
                "score": 0.4,
                "scorable": True,
                "severity": -0.1,
            }
        ]
    )
    frame.to_parquet(live_dir / "anomaly_scores.parquet", index=False)


def _write_anomaly_events(live_dir: Path, *, alpha: float = 0.05) -> None:
    frame = pd.DataFrame(
        [
            {
                "detector_id": "conformal_MSTL",
                "unique_id": "ES",
                "event_id": "evt-0",
                "alpha": alpha,
                "start_ds": pd.Timestamp("2024-08-07T02:00:00"),
                "end_ds": pd.Timestamp("2024-08-07T03:00:00"),
                "n_points": 2,
                "duration_steps": 2,
                "peak_score": 1.8,
                "peak_severity": 0.6,
                "cum_severity": 1.0,
                "peak_ds": pd.Timestamp("2024-08-07T02:00:00"),
                "direction": "up",
            }
        ]
    )
    frame.to_parquet(live_dir / "anomaly_events.parquet", index=False)


def _seed_full_refresh(live_dir: Path, *, generated_at: datetime, alpha: float = 0.05) -> None:
    _write_manifest(live_dir, generated_at=generated_at, alpha=alpha)
    _write_forecasts(live_dir)
    _write_windows(live_dir)
    _write_anomaly_scores(live_dir)
    _write_anomaly_events(live_dir, alpha=alpha)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """`TestClient` con `_get_live_dir` todavia sin sustituir por cada test."""
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _override_live_dir(live_dir: Path) -> None:
    app.dependency_overrides[_get_live_dir] = lambda: live_dir


class TestHealth:
    def test_sin_refresco_es_unavailable(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "unavailable"
        assert body["generated_at"] is None
        assert body["live_artifacts"] == {
            "forecasts": False,
            "windows": False,
            "anomaly_scores": False,
            "anomaly_events": False,
        }

    def test_refresco_reciente_es_ok(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC) - timedelta(minutes=5))
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["run_id"] == "01J9ZQK8N3F5XW1R7T2Y6H4B0C"
        assert all(body["live_artifacts"].values())

    def test_refresco_viejo_es_stale(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC) - timedelta(hours=13))
        response = client.get("/health")
        assert response.json()["status"] == "stale"


class TestModels:
    def test_lista_los_modelos_del_ultimo_refresco(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC))
        response = client.get("/models")
        assert response.status_code == 200
        model_ids = {entry["model_id"] for entry in response.json()["models"]}
        assert model_ids == {MODEL_A, MODEL_B}

    def test_404_sin_refresco(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        response = client.get("/models")
        assert response.status_code == 404


class TestForecast:
    def test_devuelve_intervalos_para_la_serie_pedida(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC))
        response = client.post("/forecast", json={"unique_id": "ES", "horizon": 6})
        assert response.status_code == 200
        body = response.json()
        assert body["unique_id"] == "ES"
        model_ids = {m["model_id"] for m in body["models"]}
        assert model_ids == {MODEL_A, MODEL_B}
        first_model = next(m for m in body["models"] if m["model_id"] == MODEL_A)
        assert [p["h_step"] for p in first_model["points"]] == [1, 2, 3]
        assert first_model["points"][0]["quantiles"] == {
            "0.025": 96.0,
            "0.5": 101.0,
            "0.975": 106.0,
        }

    def test_recorta_al_horizonte_pedido(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC))
        response = client.post("/forecast", json={"unique_id": "ES", "horizon": 2})
        body = response.json()
        first_model = body["models"][0]
        assert [p["h_step"] for p in first_model["points"]] == [1, 2]

    def test_filtra_por_los_modelos_pedidos(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC))
        response = client.post(
            "/forecast", json={"unique_id": "ES", "horizon": 6, "models": [MODEL_A]}
        )
        body = response.json()
        assert [m["model_id"] for m in body["models"]] == [MODEL_A]

    def test_404_serie_desconocida(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC))
        response = client.post("/forecast", json={"unique_id": "NOPE", "horizon": 6})
        assert response.status_code == 404

    def test_404_modelo_desconocido(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC))
        response = client.post(
            "/forecast", json={"unique_id": "ES", "horizon": 6, "models": ["Ausente"]}
        )
        assert response.status_code == 404

    def test_404_sin_refresco(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        response = client.post("/forecast", json={"unique_id": "ES", "horizon": 6})
        assert response.status_code == 404


class TestAnomalies:
    def test_devuelve_eventos_que_solapan_el_rango(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC), alpha=0.05)
        response = client.post(
            "/anomalies",
            json={
                "unique_id": "ES",
                "start": "2024-08-07T00:00:00",
                "end": "2024-08-07T06:00:00",
                "alpha": 0.05,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["events"]) == 1
        assert body["events"][0]["event_id"] == "evt-0"

    def test_rango_sin_solape_no_trae_eventos(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC), alpha=0.05)
        response = client.post(
            "/anomalies",
            json={
                "unique_id": "ES",
                "start": "2024-08-01T00:00:00",
                "end": "2024-08-01T06:00:00",
                "alpha": 0.05,
            },
        )
        assert response.status_code == 200
        assert response.json()["events"] == []

    def test_422_si_el_alpha_no_coincide_con_el_del_refresco(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _override_live_dir(tmp_path)
        _seed_full_refresh(tmp_path, generated_at=datetime.now(UTC), alpha=0.05)
        response = client.post(
            "/anomalies",
            json={
                "unique_id": "ES",
                "start": "2024-08-07T00:00:00",
                "end": "2024-08-07T06:00:00",
                "alpha": 0.10,
            },
        )
        assert response.status_code == 422
        assert "0.05" in response.json()["detail"]

    def test_404_sin_refresco(self, client: TestClient, tmp_path: Path) -> None:
        _override_live_dir(tmp_path)
        response = client.post(
            "/anomalies",
            json={
                "unique_id": "ES",
                "start": "2024-08-07T00:00:00",
                "end": "2024-08-07T06:00:00",
                "alpha": 0.05,
            },
        )
        assert response.status_code == 404


class TestOpenAPI:
    def test_documenta_las_cuatro_rutas_con_ejemplos(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == {"/health", "/models", "/forecast", "/anomalies"}

        forecast_body = schema["paths"]["/forecast"]["post"]["requestBody"]
        ref = forecast_body["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[-1]
        assert "examples" in schema["components"]["schemas"][ref]
