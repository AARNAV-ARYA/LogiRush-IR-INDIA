"""Run from the backend directory. Outputs use a frozen corpus and provisional judgments."""
import csv
import hashlib
import json
import os
from datetime import date
from functools import lru_cache
from math import log2
from pathlib import Path
from statistics import mean
from time import perf_counter

from src.ir.corpus import build_corpus
from src.ir.engine import IREngine
from src.ir.evaluation.dataset import get_queries
from src.ir.query.parser import parse_query
from src.ir.query.expansion import expand_query
from src.ir.fusion import METHODS
from src.ir.schemas import IRDocument

OUT = Path(__file__).parent / "experiment"
SNAPSHOT = OUT / "corpus.json"
AS_OF = date(2026, 11, 20)
EXTRA = [
    ("Which reports describe one-way reopening in Ramban in April 2025?", "RAM-23"),
    ("Sanklang bridge access in northern Sikkim in 2024", "SAN-15"),
    ("Air freight disruptions from Fengal in Tamil Nadu", "MAA-30"),
    ("Flight disruption in Odisha during Dana", "BBI-24"),
    ("Which advisory mentions Tripura bridge load restrictions?", "DOC_A002"),
    ("Submerged highway near Muzaffarpur", "DOC_F004"),
    ("Manali Leh corridor blockage following landslides", "DOC_L003"),
    ("Monsoon risks along NH66 in Kerala", "DOC_F005"),
    ("Where are heat damaged road surfaces on NH48?", "DOC_H001"),
    ("Cold chain guidance for medicine in the northeast", "DOC_B003"),
    ("Which report addresses the coastal highway NH16 cyclone risk?", "DOC_C002"),
    ("Accident affecting traffic near Haridwar", "DOC_A001"),
]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def metrics(ids, grades):
    def gain(values):
        return sum((2**g-1)/log2(i+2) for i,g in enumerate(values))
    ideal = gain(sorted(grades.values(), reverse=True)[:10])
    return dict(ndcg10=gain([grades.get(i, 0) for i in ids[:10]])/ideal if ideal else 0,
                precision5=sum(grades.get(i, 0)>0 for i in ids[:5])/5,
                known_recall10=sum(grades.get(i,0)>0 for i in ids[:10])/max(1,sum(v>0 for v in grades.values())))


def main():
    from dataclasses import asdict
    if not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps([asdict(d) for d in build_corpus()], indent=1))
    docs = [IRDocument(**row) for row in json.loads(SNAPSHOT.read_text())]
    engine = IREngine().build(documents=docs)
    engine._hybrid.reference_date = AS_OF
    queries = get_queries() + [dict(id=f"E{i:02}",query=q,relevant_documents=[doc]) for i,(q,doc) in enumerate(EXTRA,1)]
    judgments = {q["id"]: {i:1 for i in q["relevant_documents"]} for q in queries}
    manual = OUT / "adjudicated.csv"
    if manual.exists():
        for row in csv.DictReader(manual.open()):
            grade = int(row["grade"])
            if grade not in (0,1,2): raise ValueError("grade must be 0, 1 or 2")
            judgments[row["query_id"]][row["document_id"]] = grade
    runs, pool = [], []
    for q in queries:
        pq = parse_query(q["query"])
        pq.expanded_terms = expand_query(pq.raw,pq.hazard,pq.cargo)
        grades = judgments[q["id"]]
        # Hold out all archive events and their paraphrases as one group.
        split = "archive test" if any(i.startswith(("RAM-", "SAN-", "BBI-", "MAA-")) for i in q["relevant_documents"]) else "development"
        candidates = set(grades)
        for method in ["bm25","dense","hybrid","rrf","context_rrf","diverse_rrf","without_geo","without_time","without_source","feedback"]:
            if method == "dense" and not engine._dense.is_ready: continue
            start = perf_counter()
            if method == "feedback":
                initial = engine.fusion.search(pq,10,"context_rrf", reference=AS_OF)
                seen = [r.document.id for r in initial.results[:2]]
                # Only explicit judgments can be used as negative feedback.
                fb = {"relevant":[i for i in seen if grades.get(i,0)>0],
                      "irrelevant":[i for i in seen if grades.get(i)==0]}
                result = engine.fusion.search(pq,10,"context_rrf",feedback=fb,reference=AS_OF)
                ids = [r.document.id for r in result.results]
            elif method in METHODS:
                ids = [r.document.id for r in engine.fusion.search(pq,10,method,reference=AS_OF).results]
            else:
                ids = [i for i,_ in engine._hybrid.retrieve_by_method(pq,method,10)]
            runs.append(dict(query_id=q["id"], query=pq.raw,split=split,method=method,
                             **metrics(ids,grades),latency_ms=(perf_counter()-start)*1000,ranking=" ".join(ids)))
            candidates.update(ids)
        pool.extend(dict(query_id=q["id"],query=pq.raw,document_id=i,grade="") for i in sorted(candidates))
    write_csv(OUT/"rankings.csv",runs)
    write_csv(OUT/"annotation_pool.csv",pool)
    groups = sorted({(r["split"],r["method"]) for r in runs})
    summary = []
    for split,method in groups:
        rows = [r for r in runs if (r["split"],r["method"])==(split,method)]
        summary.append(dict(split=split,method=method,queries=len(rows),**{
            k:round(mean(r[k] for r in rows),4) for k in ("ndcg10","precision5","known_recall10","latency_ms")}))
    write_csv(OUT/"summary.csv",summary)
    (OUT/"judgments.json").write_text(json.dumps(judgments,indent=1))
    (OUT/"manifest.json").write_text(json.dumps(dict(baseline="6ad14bc",corpus_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        documents=len(docs), queries=len(queries),dense=engine._dense.is_ready,as_of=str(AS_OF)),indent=2))

if __name__=="__main__":
    main()
