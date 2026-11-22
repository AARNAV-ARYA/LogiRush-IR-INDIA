# Reproducing the final experiment

Start in routeOptimiserBackend. Install the normal requirements for the Flask app. For a CPU-only retrieval experiment use numpy, scikit-learn, onnxruntime, tokenizers and the backend dependencies. To reproduce the recorded model path download onnx/model.onnx and tokenizer.json from sentence-transformers/all-MiniLM-L6-v2 on Hugging Face into one directory. Check their hashes against encoder.sha256. Use Python 3.12.

    LOGIRUSH_MINILM_DIR=/path/to/encoder PYTHONHASHSEED=0 python run_final_benchmark.py
    NER_LIVE_WEATHER=0 python -m pytest -q

Without LOGIRUSH_MINILM_DIR the normal sentence-transformers configuration applies. Check manifest.json for dense availability. The snapshot excludes live database writes. The fixed date is 2026-11-20. The baseline uses the original hybrid score formula from commit 6ad14bc on the shared snapshot, with deterministic tie handling in the hybrid output. Latencies depend on hardware. The recorded tests use development dependencies and offline weather.

## Independent annotation

Give each reviewer a separate copy of annotation_pool.csv and corpus.json. Label each candidate 2 if it directly answers the requested event, place and time, 1 if useful but partial, and 0 if irrelevant or superseded for that question. Compare the independently completed sheets by query_id and document_id. Resolve differences together. Save the agreed rows as adjudicated.csv and rerun. Unknown judgments currently contribute zero gain. Recall measures coverage of known positives only. An adjudicated file overrides provisional grades. Some relevant reports may lie outside the pool.

For claim review, use claim_audit.csv. Review each claim against the linked original passage, date, jurisdiction and vehicle class. Enter 1 or 0 in supported for each answered claim and leave abstentions blank. Run python audit_claims.py from this directory for supported-claim rate, coverage and unsupported-answer rate. An unanswered question contributes to coverage only. No independent ratings are supplied with this snapshot.

## Three demonstrations

1. Under Historical disruption reports, search Ramban through April 22 and April 23, 2025. Compare available updates. In Intelligence Search select adaptive RRF and require NH44 and a publication cutoff.
2. Search transport disruption Assam with adaptive RRF. Mark a bridge or flood report relevant and another result not relevant. Select Rerank with feedback and inspect the terms and ranking. Explicit metadata filters persist.
3. Ask whether a truck can use NH44 in Ramban today. Current access is unverified. Inspect the evidence and abstention.

The optional classifier, cross-encoder and hyperlink centrality experiments were outside this implementation. Incident groups are curated. Time filtering uses publication dates. The historical scoring fallback uses publication year if event time is unknown. Daily records cannot order same-day updates.
