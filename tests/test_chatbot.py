
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def mock_groq(monkeypatch):
    """Replace the real Groq client with a controllable mock."""
    create = AsyncMock()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )

    monkeypatch.setattr(main, "client", fake_client)
    return create


# ---------- Unit / route behavior tests ----------

def test_chat_success(client, mock_groq):
    mock_groq.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello! How can I help?"
                )
            )
        ]
    )

    response = client.post(
        "/chat",
        json={"message": "Hello"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Hello! How can I help?"
    }

    mock_groq.assert_awaited_once()


def test_chat_sends_message_to_groq(client, mock_groq):
    mock_groq.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Test reply")
            )
        ]
    )

    response = client.post(
        "/chat",
        json={"message": "Explain AWS"},
    )

    assert response.status_code == 200

    call_args = mock_groq.await_args
    assert call_args.kwargs["messages"] == [
        {
            "role": "user",
            "content": "Explain AWS",
        }
    ]


def test_groq_failure_returns_502(client, mock_groq):
    mock_groq.side_effect = Exception("Simulated provider failure")

    response = client.post(
        "/chat",
        json={"message": "Hello"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "AI service request failed. Please try again."
    )


def test_missing_groq_client_returns_503(client, monkeypatch):
    monkeypatch.setattr(main, "client", None)

    response = client.post(
        "/chat",
        json={"message": "Hello"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "AI service is not configured."
    )


def test_empty_message_is_rejected(client, mock_groq):
    response = client.post(
        "/chat",
        json={"message": ""},
    )

    assert response.status_code == 422
    mock_groq.assert_not_awaited()


def test_oversized_message_is_rejected(client, mock_groq):
    response = client.post(
        "/chat",
        json={"message": "a" * 4001},
    )

    assert response.status_code == 422
    mock_groq.assert_not_awaited()


# ---------- HTTP integration tests ----------

def test_homepage_returns_html(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "My AI Chatbot" in response.text


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
