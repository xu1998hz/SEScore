from bart_score import BARTScorer
import torch
import numpy as np

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')

df = pd.read_csv('zhen-comet.csv', index_col=False)
df = df[["ref", "mt", "score"]]
ref_ls = df["ref"].astype(str).tolist()
mt_ls = df["mt"].astype(str).tolist()
score_ls = df["score"].astype(float).tolist()

segscores_ls = []

batch_size = 64

for mt_line, ref_line in zip(batchify(mt_ls, batch_size), batchify(ref_ls, batch_size)):
    cur_score = bart_scorer.score(mt_line, ref_line, batch_size=batch_size)
    segscores_ls.extend(cur_score)
    print("Finish one batch!")

scores_arr = np.array(score_ls)
segscores_arr = np.array(segscores_ls)


print(stats.pearsonr(segscores_arr, scores_arr))
print(stats.spearmanr(segscores_arr, scores_arr))
