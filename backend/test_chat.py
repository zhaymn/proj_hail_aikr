from fastapi.testclient import TestClient

from research_agent.api import app, runtime
from research_agent.schemas import ChatResponse, Mode


client = TestClient(app)


def test_chat_endpoint_returns_runtime_response(monkeypatch) -> None:
    expected = ChatResponse(
        session_id="test-session",
        mode=Mode.GLOBAL,
        answer="Backend smoke test response.",
        citations=[],
        debug={"smoke": True},
    )

    def fake_chat(request):
        assert request.session_id == "test-session"
        assert request.mode == Mode.GLOBAL
        assert request.message == "Say hi."
        return expected

    monkeypatch.setattr(runtime, "chat", fake_chat)

    response = client.post(
        "/api/chat",
        json={
            "session_id": "test-session",
            "mode": "global",
            "message": "Say hi.",
            "paper_ids": [],
            "history": [],
        },
    )

    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json")
