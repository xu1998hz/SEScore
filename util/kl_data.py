import csv
import pandas as pd
import json
# kl divergence according to the definition (log base 2)
def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def check_score_range(score):
    for i in range(len(range(0, -26, -1))-1):
        upper, lower = tuple(range(0, -26, -1))[i:i+2]
        if score < upper and score >= lower:
            return lower

mqmScoreFile = open('mqm_newstest2020_zhen.avg_seg_scores.tsv', 'r')
mqmScoreReader = csv.reader(mqmScoreFile, delimiter="\t")
header = next(mqmScoreReader)

mqmScores = []
for row in mqmScoreFile:
    row = row.split()
    msystem, mscore, mid = row[0], row[1], row[2]
    mqmScores.append(float(mscore))

# save all the data in the json format
saveFile = open('dist.json', 'w')
file_dist = {}
# load in all the files:
for noise_planner in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
    for del_noise_lam in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        for mask_noise_lam in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5]:
            file_name = f'generated_num_{noise_planner}_del_{del_noise_lam}_mask_{mask_noise_lam}.csv'
            # define all the score ranges from 0~-1 to -24~-25
            score_range_counts = [0] * 25
            df = pd.read_csv(file_name)
            df = df[["src", "mt", "ref", "score"]]
            df["src"] = df["src"].astype(str)
            df["mt"] = df["mt"].astype(str)
            df["ref"] = df["ref"].astype(str)
            df["score"] = df["score"].astype(float)

            for score in df["score"]:
                if score != 0:
                    score_range_counts[check_score_range(score)] += 1

            file_dist[file_name] = score_range_counts

# save into the json format
json.dump(file_dist, saveFile)

print("All files are generated!")
