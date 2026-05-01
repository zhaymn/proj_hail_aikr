from fastapi.testclient import TestClient

from research_agent.api import app, runtime
from research_agent.schemas import PaperListResponse


client = TestClient(app)


def test_paper_catalog_endpoint_returns_runtime_payload(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "list_papers", lambda: PaperListResponse(papers=[]))

    response = client.get("/api/papers")

    assert response.status_code == 200
    assert response.json() == {"papers": []}
