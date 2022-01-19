import pandas as pd
import csv
import math

rl = False
df = pd.read_csv('ende-comet.csv')
num_count_dict = {}
count = 0
if rl:
    df = df[["src", "mt", "score"]]
    df["src"] = df["src"].astype(str)
    df["mt"] = df["mt"].astype(str)
    df["score"] = df["score"].astype(float)
    for src, mt, score in zip(df["src"], df["mt"], df["score"]):
        mt = " ".join(mt.split())
        src = " ".join(src.split())
        if mt == src and score == 0:
            count += 1

        #score = math.ceil(score)
        if score not in num_count_dict:
            num_count_dict[score] = 1
        else:
            num_count_dict[score] += 1
else:
    df = df[["src", "mt", "ref", "score"]]
    df["src"] = df["src"].astype(str)
    df["mt"] = df["mt"].astype(str)
    df["ref"] = df["ref"].astype(str)
    df["score"] = df["score"].astype(float)

    for src, mt, ref, score in zip(df["src"], df["mt"], df["ref"], df["score"]):
        mt = " ".join(mt.split())
        ref = " ".join(ref.split())
        if mt == ref and score == 0:
            count += 1

        score = math.ceil(score)
        if score not in num_count_dict:
            num_count_dict[score] = 1
        else:
            num_count_dict[score] += 1

print("ref == mt: ", count)
print(num_count_dict)
print("Minimum value: ", min(num_count_dict.values()))
total = sum(num_count_dict.values())
new_num_count_dict = {}
for key, value in num_count_dict.items():
    new_num_count_dict[key] = value / total

print(new_num_count_dict)
