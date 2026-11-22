"""Reciprocal rank fusion, metadata constraints, Rocchio feedback and MMR."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ir.ranking.authority import authority_score
from src.ir.ranking.freshness import freshness_score
from src.ir.schemas import EvidenceBundle, RetrievedResult

METHODS = ("rrf", "context_rrf", "diverse_rrf", "without_geo", "without_time", "without_source")
FIELDS = ("state", "hazard", "source", "road", "district", "after", "before")


def roads(text):
    return set(re.findall(r"\bnh[ -]*(\d+[a-z]?)\b", text.casefold()))


def validate(filters, feedback):
    if not isinstance(filters, dict) or set(filters) - set(FIELDS):
        raise ValueError("Unknown metadata filter.")
    for key, value in filters.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid {key} filter.")
        if key in ("after", "before"):
            date.fromisoformat(value)
        if key == "road" and not roads(value):
            raise ValueError("Enter a highway such as NH44.")
    if filters.get("after", "") > filters.get("before", "9999"):
        raise ValueError("Invalid publication date range.")
    if not isinstance(feedback, dict) or set(feedback) - {"relevant", "irrelevant"}:
        raise ValueError("Invalid feedback.")
    for ids in feedback.values():
        if not isinstance(ids, list) or len(ids) > 50 or any(not isinstance(i, str) for i in ids):
            raise ValueError("Feedback must contain at most 50 document IDs per label.")
    if set(feedback.get("relevant", [])) & set(feedback.get("irrelevant", [])):
        raise ValueError("Conflicting feedback labels.")


def matches(doc, filters):
    for key, value in filters.items():
        if key == "road":
            if not roads(value) & roads(doc.highway or doc.route or ""):
                return False
        elif key in ("after", "before"):
            if not doc.date or (key == "after" and doc.date < value) or (key == "before" and doc.date > value):
                return False
        elif (getattr(doc, key) or "").casefold() != value.casefold():
            return False
    return True


class Fusion:
    def __init__(self, docs, bm25, dense):
        self.docs = list(docs)
        self.positions = {doc.id: i for i, doc in enumerate(self.docs)}
        self.bm25, self.dense = bm25, dense
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(d.title + " " + d.text for d in self.docs)

    def retrieve_by_method(self, pq, method, top_k):
        bundle = self.search(pq, top_k, method)
        return [(r.document.id, r.final_score) for r in bundle.results]

    def search(self, pq, k, method, filters=None, feedback=None, reference=None):
        start = time.perf_counter()
        filters = {} if filters is None else filters
        feedback = {} if feedback is None else feedback
        validate(filters, feedback)
        allowed = {d.id for d in self.docs if matches(d, filters)}
        text = pq.raw + " " + " ".join(pq.expanded_terms)
        lists = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense = pool.submit(self.dense.query, text, len(self.docs))
            lexical = self.bm25.normalised_query(text, top_k=len(self.docs))
            lists = [lexical, dense.result()]
        query_vec = self.vectorizer.transform([pq.raw]).toarray()[0]
        original = query_vec.copy()
        for label, weight in (("relevant", .75), ("irrelevant", -.25)):
            ids = sorted(set(feedback.get(label, [])))
            if any(i not in allowed for i in ids):
                raise ValueError("Feedback document outside the active search constraints.")
            if ids:
                query_vec += weight * np.asarray(self.matrix[[self.positions[i] for i in ids]].mean(axis=0)).ravel()
        terms = []
        if any(feedback.values()):
            query_vec = np.maximum(query_vec, 0)
            similarities = self.matrix @ query_vec
            lists.append([(self.docs[i].id, similarities[i]) for i in np.argsort(-similarities, kind="stable") if similarities[i] > 0])
            vocabulary = self.vectorizer.get_feature_names_out()
            terms = [str(vocabulary[i]) for i in np.argsort(original-query_vec) if query_vec[i] > original[i] and original[i] == 0][:8]
        query_roads = roads(pq.raw)
        q = pq.raw.casefold()
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", q))
        current = bool(re.search(r"\b(latest|recent|current|today|now)\b", q))
        weights = [1, 1, 1]
        if method != "rrf":
            exact = query_roads or any(d.district and d.district.casefold() in q for d in self.docs)
            weights = [1.6, .6, 1.5] if exact else [1, 1.4, 1.5]
        scores, components = {}, {}
        for m, ranking in enumerate(lists):
            rank = 0
            for doc_id, signal in ranking:
                if doc_id not in allowed:
                    continue
                rank += 1
                if rank > max(50, k * 3):
                    break
                scores[doc_id] = scores.get(doc_id, 0) + weights[m] / (60 + rank)
                components.setdefault(doc_id, [0., 0., 0.])[m] = float(signal)
        results = []
        for doc_id, score in scores.items():
            d = self.docs[self.positions[doc_id]]
            geo = float(bool(query_roads & roads(d.highway or d.route or "")) or any(
                name and name.casefold() in q for name in (d.state, d.district, d.location)))
            temporal = float((d.event_time or d.date or "")[:4] in years) if years else freshness_score(d.date, reference)
            authority = authority_score(d.source_type)
            if method != "rrf":
                score *= (1 + .5 * geo * (method != "without_geo")
                          + .35 * temporal * bool(current or years) * (method != "without_time")
                          + .1 * authority * (method != "without_source")
                          + .15 * bool(query_roads & roads(d.title)))
            bm, sem, tf = components[doc_id]
            results.append(RetrievedResult(d, 0, bm, tf, sem, temporal, geo, authority, score))
        results.sort(key=lambda r: (-r.final_score, r.document.id))
        if method == "diverse_rrf" and results:
            positions = [self.positions[r.document.id] for r in results]
            sims = (self.matrix[positions] @ self.matrix[positions].T).toarray()
            remaining = list(range(len(results)))
            selected = []
            peak = results[0].final_score
            while remaining and len(selected) < k:
                def utility(i):
                    redundancy = max((sims[i,j] for j in selected), default=0)
                    return .75 * results[i].final_score / peak - .25 * redundancy
                best = max(remaining, key=utility)
                selected.append(best)
                remaining = [i for i in remaining if i != best and sims[i,best] < .94]
            results = [results[i] for i in selected]
        for rank, r in enumerate(results[:k], 1):
            r.rank = rank
        parsed = dict(pq.to_dict(), filters=filters, road_ids=sorted(query_roads), years=sorted(years),
                      feedback_terms=terms, dense_available=self.dense.is_ready)
        return EvidenceBundle(pq.raw, parsed, results[:k], method, (time.perf_counter()-start)*1000)
