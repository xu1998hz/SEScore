import pyter
import click
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv('ende-comet.csv', index_col=False)
df = df[["ref", "mt", "score"]]
ref_ls = df["ref"].astype(str).tolist()
mt_ls = df["mt"].astype(str).tolist()
score_ls = df["score"].astype(float).tolist()

scores = []

for line, ref_line in zip(mt_ls, ref_ls):
    score = pyter.ter(line.split(), ref_line.split())
    scores.append(score)

scores_arr = np.array(scores)

print(np.array(score_ls).shape)
print(scores_arr.shape)
print(stats.pearsonr(np.array(score_ls), scores_arr))
print(stats.spearmanr(np.array(score_ls), scores_arr))
