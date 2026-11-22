"""Summarize independently annotated question and claim outcomes."""
from pathlib import Path
import csv
from collections import defaultdict

rows = list(csv.DictReader(Path(__file__).with_name("claim_audit.csv").open()))
answers = defaultdict(list)
for row in rows:
    answers[row["question"]].append(row)
answered = [group for group in answers.values() if group[0]["abstained"] == "0"]
labels = [r["supported"] for group in answered for r in group]
print("Coverage", len(answered) / len(answers) if answers else "unavailable")
if not labels or any(v not in {"0","1"} for v in labels):
    print("Independent claim judgments pending")
else:
    print("Supported-claim rate", labels.count("1")/len(labels))
    print("Unsupported-answer rate among answered", sum(any(r["supported"]=="0" for r in group) for group in answered)/len(answered))
