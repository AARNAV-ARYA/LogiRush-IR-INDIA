from src.ir.engine import IREngine
from src.ir.events import REPORTS
from src.ir.verification import verify
import csv,time
engine=IREngine().build()
with open('experiment/claim_audit.csv','w') as out:
 w=csv.writer(out);w.writerow(['question','abstained','claim','document_id','source_url','supported','retrieval_ms','generation_ms'])
 for q in ['Ramban NH44 April 2025','Sanklang bridge collapse in 2024','Chennai Fengal cancellations 2024','Is NH44 open near Ramban today?','Can we send a truck to Sikkim today?','Bhubaneswar Cyclone Dana 2024']:
  b=engine.search(q,method='diverse_rrf');t=time.perf_counter();a=verify(b,REPORTS);ms=(time.perf_counter()-t)*1000
  if a['abstained']:w.writerow([q,1,a['answer'],'','','',b.latency_ms,ms])
  else:
   for doc in [engine.get_document(i) for i in a['evidence_used']]:
    for claim in [doc.title]+[s.strip() for s in doc.text.split('. ') if s.strip()]:
     w.writerow([q,0,claim,doc.id,doc.source_url,'',b.latency_ms,ms])
