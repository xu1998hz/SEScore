import bert_score
import numpy as np
import pandas as pd
from scipy import stats

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

df = pd.read_csv('zhen-comet.csv', index_col=False)
df = df[["ref", "mt", "score"]]
ref_ls = df["ref"].astype(str).tolist()
mt_ls = df["mt"].astype(str).tolist()
score_ls = df["score"].astype(float).tolist()


scores = []
scorer = bert_score.scorer.BERTScorer(model_type='bert-base-multilingual-cased')

for ref_batch, mt_batch, _ in zip(batchify(ref_ls, 1024), batchify(mt_ls, 1024), batchify(score_ls, 1024)):
    print(mt_batch)
    print(ref_batch)
    _, _, cur_score = scorer.score(mt_batch, ref_batch, verbose=False)
    scores.extend(cur_score)

scores_arr = np.array(scores)

print(np.array(score_ls).shape)
print(scores_arr.shape)
print(stats.pearsonr(np.array(score_ls), scores_arr))
print(stats.spearmanr(np.array(score_ls), scores_arr))
