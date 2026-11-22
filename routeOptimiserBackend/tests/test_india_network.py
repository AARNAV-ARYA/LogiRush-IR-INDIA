import pytest
from src.services.routing_service import routing_service
from src.data_processing.ner_data_provider import MockNERDataProvider
from src.services.places_service import LocalPlaceProvider
from src.ir.events import timelines
from datetime import date


@pytest.mark.parametrize("a,b", [("Mumbai","Bengaluru"),("Delhi","Guwahati"),("Kochi","Leh"),("Ranchi","Chennai")])
def test_mainland_journeys(a,b):
    nodes = {x.name:x.id for x in MockNERDataProvider().get_locations()}
    plan = routing_service.plan(nodes[a],nodes[b],max_routes=1)
    assert plan["recommended_route"]
    assert plan["routes"]


def test_coverage_and_offshore_limits():
    provider = MockNERDataProvider()
    hubs = provider.get_locations()
    assert len({x.state for x in hubs}) == 36
    ids = {x.id for x in hubs}
    assert all(e.source in ids and e.destination in ids and e.distance_km > 0 for e in provider.get_road_segments())
    places = LocalPlaceProvider(provider)
    assert not places.search("Kavaratti")[0]["routable"]
    assert not places.nearest(10.5667,72.6417)
    assert places.search("Mumbai")[0]["routable"]
    assert timelines("Gujarat",date(2024,8,28))[0][0]["source_url"]


def test_geographic_index_and_corridor_shapes():
    from src.ir.corpus import build_corpus
    from src.ir.verification import verify
    from src.ir.events import REPORTS
    from src.ir.schemas import EvidenceBundle, RetrievedResult
    docs = build_corpus()
    assert len({d.state for d in docs if d.provenance == "DIRECTORY"}) == 36
    edges = [e for e in MockNERDataProvider().get_road_segments() if e.id.startswith("IND")]
    assert len(edges) == 36
    assert all(len(e.geometry) > 2 for e in edges)
    sample = EvidenceBundle("Mumbai floods in 2024", {}, [RetrievedResult(d,1) for d in REPORTS])
    assert verify(sample,REPORTS)["abstained"]
