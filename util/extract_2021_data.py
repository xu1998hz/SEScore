import pandas as pd

df = pd.read_csv('zhen_vali_2021.csv')
df = df[["src", "mt", "ref", "score"]]
df["ref"] = df["ref"].astype(str)

ref_set = []
for ref_sen in df["ref"]:
    if ref_sen not in ref_set:
        ref_set.append(ref_sen)

src_set = []
for src_sen in df["src"]:
    if src_sen not in src_set:
        src_set.append(src_sen)

src_file = open('src_2021.txt', 'w')
ref_file = open('ref_2021.txt', 'w')

for src_sen, ref_sen in zip(src_set, ref_set):
    src_file.write(src_sen+'\n')
    ref_file.write(ref_sen+'\n')

print("generated!")
