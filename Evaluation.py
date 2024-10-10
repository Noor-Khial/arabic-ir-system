
import pandas as pd 
import pyterrier as pt
if not pt.started():
  pt.init()

qrels = pd.read_csv("data/qrels.txt", sep=',', names=['qid', 'docno', 'Q0', 'label'], header=0)
qrels['docno'] = qrels['docno'].astype(str)
mcontriever_results = "results.csv"
df_mcontriever_results= pd.read_csv(mcontriever_results)
mcontriever_evaluation = pt.Evaluate(df_mcontriever_results,qrels[['qid','docno','label']],metrics=["P", "recall","recip_rank"])

#df = pd.DataFrame([mcontriever_evaluation['R@5'], mcontriever_evaluation['recip_rank' ]], columns=["map", "R@5"])
#output_file = "data/retrieving_results/measurements_results.csv"
#df.to_csv(output_file, index=False)