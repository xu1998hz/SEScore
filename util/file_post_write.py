import pandas as pd

df = pd.read_csv('zhen-comet.csv', index_col=False)
df = df[['src', "ref", "mt", "score"]]
src_ls = df["src"].astype(str)
ref_ls = df["ref"].astype(str)
mt_ls = df["mt"].astype(str)
score_ls = df["score"].astype(float)

for src, ref, mt, score in zip(src_ls, ref_ls, mt_ls, score_ls):
    print(src) 
