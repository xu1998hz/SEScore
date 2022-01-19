import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

df = pd.read_csv('shuffle_both_paws_post_jan10_zhen_news.csv')
df = df[["src", "mt", "score"]]
df["src"] = df["src"].astype(str)
df["mt"] = df["mt"].astype(str)
df["score"] = df["score"].astype(float)

score_dict = {}

for score in df['score']:
    if score != 0:
        if score in score_dict:
            score_dict[score] += 1
        else:
            score_dict[score] = 1

scores_ls, count_ls = list(score_dict.keys()), list(score_dict.values())
scores_ls = [math.log(-score+0.01) for score in scores_ls]
count_ls = [math.log(count) for count in count_ls]
#print(count_ls)
x = np.array(scores_ls)
y = np.array(count_ls)
m, b = np.polyfit(x, y, 1)
print(m)
print(b)
plt.plot(x, m*x + b)
plt.scatter(scores_ls, count_ls)
plt.title('Score Distribution')
#plt.xlim(right=3)
plt.xlabel("Score values by log")
#plt.gca().invert_yaxis()
plt.ylabel("count by log")
plt.show()
