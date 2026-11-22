"""Extract attributable evidence for context-aware searches and expose status conflicts."""
import re
from src.ir.ranking.geographic import CITY_COORDS, STATE_CITIES
from src.ir.fusion import matches, roads
from src.ir.events import access_answer, operational_question

# Status words refer only to explicit archived observations in curated event groups.
def status(doc):
    text = doc.title.casefold()
    if re.search(r"reopen|resumed", text):
        return "reopening"
    if re.search(r"closed|closure", text):
        return "closure"
    return None


def verify(bundle, archive):
    query = bundle.query.casefold()
    highways = roads(query)
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
    names = set(CITY_COORDS) | {d.location.casefold() for d in archive}
    mentioned = {name for name in names if re.search(r"\b"+re.escape(name)+r"\b",query)}
    states = {name for name in STATE_CITIES if re.search(r"\b"+re.escape(name)+r"\b",query)}
    mentioned -= states
    anchor = highways or mentioned or any(d.state.casefold() in query for d in archive if d.state)
    docs = []
    # A matching closure can retrieve its later update even if the update itself
    # fell below top K. Apply the same explicit constraints to every linked report.
    candidates = {r.document.id: r.document for r in bundle.results}
    event_ids = {d.event_id for d in candidates.values() if d.event_id}
    filters = bundle.parsed.get("filters", {})
    candidates.update({d.id:d for d in archive if d.event_id in event_ids and matches(d, filters)})
    explicit_days = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
    for d in candidates.values():
        if explicit_days and d.date > max(explicit_days):
            continue
        if not d.source_url or not d.event_id:
            continue
        if highways and not highways.intersection(roads(d.highway or "")):
            continue
        if mentioned and d.location.casefold() not in mentioned:
            continue
        if states and (d.state or "").casefold() not in states:
            continue
        if years and (d.event_time or d.date)[:4] not in years:
            continue
        if not anchor and not set(re.findall(r"\w{4,}",query)) & set(re.findall(r"\w{4,}",d.title.casefold())):
            continue
        docs.append(d)
    live = operational_question(query) and (not years or bool(re.search(r"\b(today|now|currently)\b",query)))
    if live or not docs:
        answer = access_answer(bundle)
        answer.update(answer="Insufficient attributable evidence to confirm the requested condition.",
                      evidence_used=[], note="Search results remain available for inspection.")
        return answer
    events = {}
    for d in docs:
        if status(d):
            events.setdefault((d.event_id,d.event_time or d.date),set()).add(status(d))
    conflict = any(len(values)>1 for values in events.values())
    ordered = sorted(docs, key=lambda d:(d.event_time or d.date,d.id))[-5:]
    return dict(answer="\n".join(f"{d.date} · {d.title}. {d.text} [{d.id}]" for d in ordered),
                provider="archived passages", risk_level="UNKNOWN", grounded=True,
                evidence_used=[d.id for d in ordered], abstained=conflict,
                note="Conflicting status reports; passage is unconfirmed." if conflict else
                     "Dated source summaries. Later observations can revise earlier conditions.")
