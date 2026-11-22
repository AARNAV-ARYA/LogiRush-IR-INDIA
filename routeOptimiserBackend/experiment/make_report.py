"""Generate the report from recorded runs. Requires reportlab and matplotlib."""
import csv
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Image

root = Path(__file__).parent
rows = list(csv.DictReader((root/"summary.csv").open()))
style = getSampleStyleSheet()
story = []
def add(text, label="BodyText"):
    story.extend([Paragraph(text, style[label]), Spacer(1,6)])
def table(data):
    t = Table(data,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16465a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#ecf4f3")]),("FONTSIZE",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,12)])
add("LogiRush", "Title")
add("An evidence-grounded, adaptive information retrieval system for Pan-India multi-hazard logistics intelligence","Heading2")
add("CSD358 · Final project implementation and experimental results · November 2026")
add("Research question","Heading2")
add("Can query-aware rank fusion retrieve the correct corridor and dated disruption evidence more reliably than fixed hybrid ranking? We compare the original scoring formula, standard reciprocal rank fusion and a rule-based extension on identical documents.")
add("Retrieval design","Heading2")
add("BM25 and MiniLM produce candidate rankings concurrently. Standard RRF sums 1/(60 + rank) over contributing lists. Adaptive fusion gives lexical retrieval greater weight for explicit roads and districts, and semantic retrieval greater weight for broad descriptions. Geographic agreement, source prior, title-road agreement and temporal intent modify the fused score.")
add("State, district, hazard, source, highway and inclusive publication-date bounds can be mandatory. Inferred metadata affects ranking. Documents with unknown required values cannot satisfy the constraint. The historical signal uses event year where available and publication year otherwise. Publication date, event time, collection time and expiry remain separate schema fields.")
add("Interactive retrieval","Heading2")
add("Rocchio updates a TF-IDF query vector with 0.75 times the positive centroid and subtracts 0.25 times the negative centroid. A feedback ranking enters fusion and the UI shows the strongest new terms. Explicit filters constrain each round. MMR greedily balances normalized relevance and maximum selected-document similarity at 0.75/0.25; cosine similarity of at least 0.94 suppresses near copies.")
add("Evidence handling","Heading2")
add("The new fusion modes return dated, attributable archive summaries or abstain. Explicit roads, recognized archive places and years constrain the eligible evidence. Conflicting closure and reopening labels on the same event-day trigger abstention. A separate timeline preserves updates across days. Current vehicle passage requires evidence beyond historical reports. The original RAG path remains available for the baseline methods.")
add('<link href="https://github.com/AARNAV-ARYA/LogiRush-IR" color="blue">Base GitHub repository</link> · Mid-semester reference commit 6ad14bc. The accompanying ZIP contains the extension.')
story.append(PageBreak())
add("Retrieval experiment","Title")
add("The frozen corpus contains 1,028 documents, including 1,000 generated examples and five externally sourced archive reports. The evaluation uses 33 development requests and eight archive requests covering four events. The benchmark includes paraphrases; eight requests do not imply eight independent incidents. Labels are provisional binary positives. The scoring script also accepts adjudicated grades of 0, 1 and 2.")
methods=["bm25","dense","hybrid","rrf","context_rrf","diverse_rrf","feedback"]
labels=["BM25","Dense MiniLM","Original hybrid","Standard RRF","Adaptive RRF","Adaptive + MMR","Simulated feedback"]
def row(split,method):return next(r for r in rows if r["split"]==split and r["method"]==method)
table([["Archive queries","nDCG@10","P@5","Known recall@10","Search ms"]]+[[label]+[f'{float(row("archive test",m)[k]):.3f}' for k in ["ndcg10","precision5","known_recall10","latency_ms"]] for m,label in zip(methods,labels)])
fig,ax=plt.subplots(figsize=(7.1,2.6));x=list(range(6))
for offset,split,color in [(-.18,"development","#245b75"),(.18,"archive test","#c36e34")]:
    ax.bar([i+offset for i in x],[float(row(split,m)["ndcg10"]) for m in methods[:6]],width=.35,label=split,color=color)
ax.set_xticks(x,["BM25","Dense","Hybrid","RRF","Adaptive","+ MMR"]);ax.set_ylim(0,1.1);ax.set_ylabel("nDCG@10");ax.legend(ncols=2,frameon=False);fig.tight_layout()
buf=BytesIO();fig.savefig(buf,format="png",dpi=160);buf.seek(0);story.append(Image(buf, width=450,height=165))
add("Interpretation","Heading2")
add("On the archive split, adaptive fusion with diversity recovered every known positive within ten results. BM25 ranked the first relevant reports more strongly on average. On development queries BM25 also led on nDCG. These results support an archive-specific benefit over fixed fusion and do not establish consistent superiority across the full task.")
add("Simulated feedback considers the first two results using existing judgments. Missing labels are never treated as user dislikes. Inspect rankings.csv for per-query failures and changed ranks. Feedback latency includes both passes.")
story.append(PageBreak())
add("Ablations and evidence audit","Title")
table([["Ranker","Development nDCG@10","Archive nDCG@10"]]+[[m]+[row(s,m)["ndcg10"] for s in ["development","archive test"]] for m in ["context_rrf","without_geo","without_time","without_source"]])
add("Removing geographic context reduced archive performance. Removing the temporal feature made no change to aggregate nDCG in this experiment. Incomplete judgments, small event counts and limited publication periods prevent a strong conclusion about time weighting.")
add("Verification and remaining work","Heading2")
add("The six-question evidence audit includes archived questions and present-access questions. Its CSV separates retrieval and generation timings and links each returned claim to the original source. Independent reviewers must mark support before a supported-claim rate or unsupported-answer rate can be reported. Current abstention is deliberately conservative and does not certify route availability.")
add("Two reviewers should grade the candidate pool independently, resolve disagreements, then rerun the experiment with adjudicated.csv. All ranked methods must use the same corpus snapshot. The states represented in the sourced pilot are Tamil Nadu, Odisha and Sikkim, plus the union territory of Jammu and Kashmir. National coverage remains a design goal.")
add("Reproduction and demonstration","Heading2")
add("experiment/README.md lists the model files, commands and three demo cases. corpus.json freezes the collection; manifest.json and encoder.sha256 identify the inputs. summary.csv and rankings.csv contain aggregate and per-query results. The complete source and dependency files accompany this report.")
add("Research connections","Heading2")
for title,url in [
("Robertson &amp; Zaragoza (2009) · BM25 lexical baseline","https://doi.org/10.1561/1500000019"),
("Cormack, Clarke &amp; Buettcher (2009) · equal-weight RRF baseline","https://doi.org/10.1145/1571941.1572114"),
("Salton &amp; Buckley (1990) · vector-space relevance feedback","https://www.cs.ucr.edu/~vagelis/classes/CS172/publications/jasistSalton1990.pdf"),
("Carbonell &amp; Goldstein (1998) · MMR for evidence diversity","https://doi.org/10.1145/290941.291025"),
("Reimers &amp; Gurevych (2019) · sentence embedding background","https://aclanthology.org/D19-1410/"),
("Es et al. (2024) · distinguish evidence support from retrieval performance","https://aclanthology.org/2024.eacl-demo.16/")]:
    add(f'<link href="{url}" color="blue">{title}</link>')
SimpleDocTemplate(str(root/"LogiRush_Final_Report.pdf"),pagesize=A4,topMargin=42,bottomMargin=42,leftMargin=50,rightMargin=50).build(story)
