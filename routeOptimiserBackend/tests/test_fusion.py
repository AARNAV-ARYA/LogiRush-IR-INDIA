from src.ir.engine import IREngine
import pytest
from src.ir.fusion import validate


def test_fusion_filters_feedback_diversity():
    engine = IREngine().build()
    for method in ("rrf", "context_rrf", "diverse_rrf"):
        result = engine.search("NH-44 closure", method=method, filters={"road": "NH 44", "before": "2025-04-22"})
        ids = [r.document.id for r in result.results]
        assert "RAM-21" in ids and "RAM-23" not in ids
        result = engine.search("NH-44 closure", method=method, filters={"road":"NH44"}, feedback={"relevant":["RAM-21"]})
        assert result.parsed["feedback_terms"]
    with pytest.raises(ValueError):
        engine.search("road", method="context_rrf", filters={"state":"Sikkim"}, feedback={"relevant":["RAM-21"]})
    with pytest.raises(ValueError):
        validate({"after":"2026-14-12"}, {})


def test_verified_historical_answer_and_present_abstention():
    from src.ir.events import REPORTS
    from src.ir.verification import verify
    engine = IREngine().build()
    history = engine.search("Ramban NH44 April 2025", method="context_rrf")
    assert "RAM-23" in verify(history, REPORTS)["evidence_used"]
    current = engine.search("Is NH44 open today?",method="diverse_rrf")
    assert verify(current, REPORTS)["abstained"]
