from fastapi.testclient import TestClient

from app.main import app


def test_liveness_and_readiness_endpoints():
    with TestClient(app) as client:
        assert client.get("/livez").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}
        assert client.get("/health").status_code == 200
