import csv
import re
import pandas as pd
from itertools import combinations

# mqmScoreFile = open('mqm_newstest2020_ende.avg_seg_scores.tsv', 'r')
# mqmScoreReader = csv.reader(mqmScoreFile, delimiter="\t")
# header = next(mqmScoreReader)
#
# seg_score_dist = {}
# for row in mqmScoreFile:
#     row = row.split()
#     msystem, mscore, mid = row[0], row[1], row[2]
#     if mid not in seg_score_dist:
#         seg_score_dist[mid] = [mscore]
#     else:
#         seg_score_dist[mid].append(mscore)


df = pd.read_csv('noise_en_de_20.csv')
df = df[["src", "mt", "ref", "score"]]
df["src"] = df["src"].astype(str)
df["mt"] = df["mt"].astype(str)
df["ref"] = df["ref"].astype(str)
df["score"] = df["score"].astype(float)

final_dict = {}
for src, mt, ref, score in zip(df["src"], df["mt"], df["ref"], df["score"]):
    if src+'<sep>'+ref not in final_dict:
        final_dict[src+'<sep>'+ref] = []
    final_dict[src+'<sep>'+ref].append([mt, float(score)])

rank_data = open("rank_ende_lastweek.csv", 'w')
csvwriter = csv.writer(rank_data)
fields = ["src", "pos", "neg", "ref"]
csvwriter.writerow(fields)
count = 0
count_err = 0
for src_ref_sen, mt_scores in final_dict.items():
    for senA, senB in list(combinations(mt_scores,2)):
        textA, scoreA = senA
        textB, scoreB = senB
        src, ref = src_ref_sen.split('<sep>')[0], src_ref_sen.split('<sep>')[1]
        #post_textA = ' '.join(re.findall(r"[\w]+|[^\s\w]", textA))
        #post_textB = ' '.join(re.findall(r"[\w]+|[^\s\w]", textB))
        if textA != textB:
            if scoreA > scoreB:
                csvwriter.writerow([src, textA, textB, ref])
            elif scoreA < scoreB:
                csvwriter.writerow([src, textB, textA, ref])
        else:
            print(textA)
            print(textB)
            if scoreA != scoreB:
                count_err += 1
            count += 1
            print('--------------------')

print("Count total: ", count)
print("Count err: ", count_err)
print('ranking model is generated!')
# # print(seg_score_dist)
# my_seg_score_dist = {}
# Score40File = open('noise_ende_200k_final.csv', 'r')
# Score40Reader = csv.reader(Score40File)
# header = next(Score40Reader)
# for row in Score40File:
#     if row.split(',')[0] not in my_seg_score_dist:
#         my_seg_score_dist[row.split(',')[0]] = [float(row.split(',')[-1][:-1])]
#     else:
#         my_seg_score_dist[row.split(',')[0]].append(float(row.split(',')[-1][:-1]))
#
# print(my_seg_score_dist)
# len_dist = []
# for key, value in my_seg_score_dist.items():
#     len_dist.append(len(value))
#
# #print(len_dist)
