"""Curated event identities and dated observations. Dates use Indian local time."""
from datetime import date
import re

from src.ir.schemas import IRDocument


def report(identifier, event, day, place, state, highway, hazard, title, summary, source, url):
    return IRDocument(
        id=identifier, event_id=event, date=day, location=place, state=state,
        highway=highway, hazard=hazard, severity="unknown", title=title, text=summary,
        source=source, source_url=url, retrieved_at="2026-11-22" if identifier in {"GJ-28","AS-24","KL-02","DL-04","MP-05","LD-01"} else "2026-11-20", source_type="verified_org", provenance="ARCHIVED",
    )


REPORTS = [
    report("MP-05", "narmadapuram-flood-2022", "2022-07-05", "Narmadapuram", "Madhya Pradesh",
           None, "flood", "Floodwater temporarily closed a central India highway",
           "A submerged section interrupted road access between Narmadapuram, Itarsi and Betul.",
           "Free Press Journal", "https://www.freepressjournal.in/bhopal/narmadapuram-national-highway-closed-due-to-heavy-rains"),
    report("LD-01", "lakshadweep-ockhi-2017", "2017-12-01", "Minicoy", "Lakshadweep",
           None, "cyclone", "Cyclone Ockhi warning for Lakshadweep",
           "A cyclone bulletin located the storm north of Minicoy. This warning does not record a ferry cancellation.",
           "Press Information Bureau", "https://www.pib.gov.in/newsite/PrintRelease.aspx?lang=2&reg=48&relid=174043"),
    report("GJ-28", "vadodara-flood-2024", "2024-08-28", "Vadodara", "Gujarat",
           None, "flood", "Flooding disrupted Vadodara rail services",
           "Waterlogged tracks in the Vadodara division led to cancellations and diversions.",
           "Times of India", "https://timesofindia.indiatimes.com/city/vadodara/56-trains-cancelled-due-to-water-on-tracks-in-vadodara/articleshow/112850344.cms"),
    report("AS-24", "haflong-slide-2025", "2025-06-24", "New Haflong", "Assam",
           None, "landslide", "Hillside debris interrupted Assam rail access",
           "Northeast Frontier Railway suspended trains on the Lumding–Badarpur section.",
           "Akashvani", "https://newsonair.gov.in/train-services-to-tripura-manipur-mizoram-suspended-after-landslides-hit-haflong-section-nfr/"),
    report("KL-02", "wayanad-slide-2024", "2024-08-02", "Wayanad", "Kerala",
           None, "landslide", "Chooralmala bridge enabled Wayanad relief access",
           "A temporary bridge allowed rescue personnel and machinery to reach the affected area.",
           "Akashvani", "https://newsonair.gov.in/kerala-death-toll-rises-to-281-over-180-injured-in-wayanad-landslides/"),
    report("DL-04", "delhi-fog-2025", "2025-01-04", "Delhi", "Delhi",
           None, "fog", "Fog delayed trains approaching Delhi",
           "Indian Railways reported delays to Delhi-bound services during low visibility.",
           "Akashvani", "https://newsonair.gov.in/several-delhi-bound-trains-delayed-due-to-dense-fog-low-visibility/"),
    report("MAA-30", "chennai-fengal-2024", "2024-11-30", "Chennai", "Tamil Nadu",
           None, "cyclone", "Cyclone disrupted Chennai flights",
           "Airport authorities reported six cancellations during the weather disruption.",
           "Akashvani", "https://newsonair.gov.in/wp-content/uploads/2024/11/TEL-1230-6.pdf"),
    report("BBI-24", "bhubaneswar-dana-2024", "2024-10-24", "Bhubaneswar", "Odisha",
           None, "cyclone", "Bhubaneswar airport cancellations during Cyclone Dana",
           "The regional bulletin reported flight cancellations at Bhubaneswar airport.",
           "Akashvani", "https://www.newsonair.gov.in/wp-content/uploads/2024/10/ita-eng-1950-10.pdf"),
    report("RAM-21", "ramban-april-2025", "2025-04-21", "Ramban", "Jammu and Kashmir",
           "NH44", "landslide", "Ramban landslides closed NH44",
           "Debris clearance was underway at several points on the Jammu–Srinagar highway.",
           "Akashvani", "https://newsonair.gov.in/jammu-srinagar-nh-blocked-by-landslides-one-way-traffic-likely-within-2-days-dc-ramban/"),
    report("RAM-23", "ramban-april-2025", "2025-04-23", "Ramban", "Jammu and Kashmir",
           "NH44", "landslide", "One-way traffic resumed on NH44",
           "An official spokesperson reported one-way reopening. This report does not establish truck-specific access.",
           "The Tribune", "https://www.tribuneindia.com/news/j-k/landslide-hit-jammu-srinagar-nh-opened-for-one-way-traffic-special-train-arranged/amp"),
    report("SAN-15", "sanklang-june-2024", "2024-06-15", "Sanklang", "Sikkim",
           None, "infrastructure failure", "Sanklang bridge collapse disrupted access",
           "The suspension bridge serving Dzongu collapsed during heavy rainfall and landslides.",
           "Akashvani", "https://newsonair.gov.in/sikkim-incessant-heavy-rainfall-causes-multiple-landslides-in-northern-part-of-state-over-1200-tourists-still-stranded/"),
]


def timelines(query, cutoff, reports=REPORTS):
    """Return archived observations available by end of the requested local day."""
    normalized = re.sub(r"\bnh[ -]*(\d+)", r"nh\1", query.casefold())
    highways = set(re.findall(r"\bnh\d+\b", normalized))
    groups = {}
    for doc in sorted(reports, key=lambda item: (item.date, item.id)):
        if date.fromisoformat(doc.date) > cutoff:
            continue
        road = (doc.highway or "").lower()
        names = (doc.location, doc.state, road)
        if highways and road not in highways:
            continue
        if not any(name and re.search(r"\b" + re.escape(name.lower()) + r"\b", normalized) for name in names):
            continue
        groups.setdefault(doc.event_id, []).append(doc.to_dict())
    return list(groups.values())


def operational_question(query):
    """Identify requests for present-day access or dispatch decisions."""
    return bool(re.search(
        r"\b(today|currently|right now|safe to|can (?:i|we|a truck)|"
        r"is .{0,65}(?:open|closed|passable|safe)|should (?:i|we) (?:send|dispatch))\b",
        query, re.I,
    ))


def access_answer(bundle):
    """A verified passage decision needs explicit validity and vehicle permissions.

    Current schemas have neither field. Return attributable observations only.
    """
    reports = [r.document for r in bundle.results if r.document.source_url]
    answer = "Current passage cannot be verified for the requested stretch and vehicle."
    if reports:
        answer += "\nArchived observations:\n" + "\n".join(
            f"{d.date}: {d.title} [{d.id}]" for d in reports[:3]
        )
    return dict(answer=answer, grounded=False, risk_level="UNKNOWN", provider="archive",
                evidence_used=[d.id for d in reports[:3]], abstained=True,
                note="Historical reports cannot confirm present access.")
