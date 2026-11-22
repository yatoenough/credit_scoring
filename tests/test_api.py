import pytest
from fastapi.testclient import TestClient

from credit_scoring.api import app

VALID = {"age": 30, "income": 50, "education": True, "work": True, "car": False}


@pytest.fixture
def client(model_path):
    with TestClient(app) as c:
        yield c


def test_valid_request(client):
    response = client.post("/score", json=VALID)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["approved"], bool)
    assert 0 <= body["default_probability"] <= 1


@pytest.mark.parametrize("override", [{"income": 0}, {"age": 10}])
def test_invalid_request(client, override):
    response = client.post("/score", json=VALID | override)

    assert response.status_code == 422
