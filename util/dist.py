import pandas as pd
import matplotlib.pyplot as plt
from comet import download_model, load_from_checkpoint
import csv


range_1 = 0
range_2 = 0
range_3 = 0
range_4 = 0
range_5 = 0
range_6 = 0
range_7 = 0
range_8 = 0
range_9 = 0
range_10 = 0
range_0_1 = 0
range_1_5 = 0
range_5_10 = 0
range_10_15 = 0
range_15_20 = 0
range_20_25 = 0

df = pd.read_csv('jan_10_zhen_news_comp_num_0.5_del_1.5_mask_1.5_xlm_mbart.csv')
df = df[["src", "mt", "ref", "score"]]
df["src"] = df["src"].astype(str)
df["mt"] = df["mt"].astype(str)
df["ref"] = df["ref"].astype(str)
df["score"] = df["score"].astype(float)

# csvfile = open('data/post_final_news_complementary_zhen_200k.csv', 'w')
# csvwriter = csv.writer(csvfile)
# fields = ['src', 'mt', 'ref', 'score']
# csvwriter.writerow(fields)

count = 0

for src, mt, ref, score in zip(df["src"], df["mt"], df["ref"], df["score"]):
    if score > -1 and score <= 0:
        range_0_1 += 1
    if score > -5 and score <= -1:
        if score == -1:
            range_1 += 1
        elif score == -2:
            range_2 += 1
        elif score == -3:
            range_3 += 1
        elif score == -4:
            range_4 += 1

            #range_1_5 += 1

    elif score > -10 and score <= -5:
        if score == -5:
            range_5 += 1
        if score == -6:
            range_6 += 1
        if score == -7:
            range_7 += 1
        if score == -8:
            range_8 += 1
        if score == -9:
            range_9 += 1

        range_5_10 += 1
    elif score > -15 and score <= -10:
        if score == -10:
            range_10 += 1
        range_10_15 += 1
    elif score > -20 and score <= -15:
        range_15_20 += 1
    else:
        range_20_25 += 1


print("-1: ", range_1)
print("-2: ", range_2)
print("-3: ", range_3)
print("-4: ", range_4)
print("-5: ", range_5)
print("-6: ", range_6)
print("-7: ", range_7)
print("-8: ", range_8)
print("-9: ", range_9)
print("-10: ", range_10)


print("range 0 ~ -1: ", range_0_1)
print("range -1 ~ -5: ", range_1_5)
print("range -5 ~ -10: ", range_5_10)
print("range -10 ~ -15: ", range_10_15)
print("range -15 ~ -20: ", range_15_20)
print("range -20 ~ -25: ", range_20_25)
