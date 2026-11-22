from datetime import date
from flask import Flask
import pytest

from src.api.ir_routes import ir_bp
from src.ir.events import timelines


@pytest.mark.parametrize("query,day,ids", [
    ("NH-44", "2025-04-20", []),
    ("NH 44", "2025-04-21", ["RAM-21"]),
    ("Ramban", "2025-04-23", ["RAM-21", "RAM-23"]),
    ("NH440", "2025-04-23", []),
    ("Sikkim", "2024-06-15", ["SAN-15"]),
])
def test_archived_observations(query, day, ids):
    groups = timelines(query, date.fromisoformat(day))
    assert [d["id"] for group in groups for d in group] == ids


def test_operational_question_abstains():
    app = Flask(__name__)
    app.register_blueprint(ir_bp)
    data = app.test_client().post("/api/rag/query", json={
        "query": "Is NH44 open near Ramban today?", "method": "adaptive",
    }).get_json()
    assert data["abstained"]
    assert data["risk_level"] == "UNKNOWN"
    assert data["route_recommendation"] is None


@pytest.mark.parametrize("endpoint", ["/api/search", "/api/rag/query"])
@pytest.mark.parametrize("payload", [
    ["query"], {"query": 42}, {"query": "Delhi", "top_k": "bad"},
    {"query": "Delhi", "top_k": -1}, {"query": "Delhi", "cargo_type": []},
])
def test_invalid_search_requests_return_a_client_error(endpoint, payload):
    app = Flask(__name__)
    app.register_blueprint(ir_bp)
    response = app.test_client().post(endpoint, json=payload)
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"
