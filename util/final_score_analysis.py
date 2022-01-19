import csv
import matplotlib.pyplot as plt
import numpy as np

"""
CUDA_VISIBLE_DEVICES=4 nohup python3 -u bachify_data.py -num_var 10 -lang de_DE -src train/ende_train/src -ref train/ende_train/ref -save noise_ende_200k_latest.csv > noise_ende_200k_latest.out 2>&1 &
"""

"""
CUDA_VISIBLE_DEVICES=5 nohup python3 -u bachify_data.py -num_var 10 -lang en_XX -src train/zhen_train/src -ref train/zhen_train/ref -save noise_zhen_200k_latest.csv > noise_zhen_200k_latest.out 2>&1 &
"""


mqmScoreFile = open('mqm_newstest2020_zhen.avg_seg_scores.tsv', 'r')
mqmScoreReader = csv.reader(mqmScoreFile, delimiter="\t")
header = next(mqmScoreReader)

mqmScores = []
zero_count, mqm_count, n1_count = 0, 0, 0
for row in mqmScoreFile:
    row = row.split()
    msystem, mscore, mid = row[0], row[1], row[2]
    if float(mscore) == 0:
        zero_count += 1
    elif float(mscore) > -1.3 and float(mscore) < -1:
        n1_count += 1
    mqm_count += 1
    mqmScores.append(float(mscore))

print('---------------MQM GT------------------')
print(f"Total: {mqm_count}")
print()

print(f"There are {zero_count} zeros")
print(zero_count/mqm_count)
print()

print(f"There are {n1_count} n1s")
print(n1_count/mqm_count)
print()
print('---------------------------------')
Score40File = open('noise_zhen_200k_final.csv', 'r')
Score40Reader = csv.reader(Score40File, delimiter=",")
header = next(Score40Reader)

Scores40 = []
zero_count, my_count, n1_count = 0, 0, 0
for row in Score40File:
    score = float(row.split(',')[-1][:-1])
    my_count+=1
    if score == 0:
        zero_count += 1
    elif score > -1.3 and score < -1:
        n1_count += 1
    Scores40.append(score)

print('---------------Synthesize------------------')
print(f"Total: {my_count}")
print()

print(f"There are {zero_count} zeros")
print(zero_count/my_count)
print()

print(f"There are {n1_count} n1s")
print(n1_count/my_count)
print()
print('---------------------------------')

nonzeromqm = []
for mqm in mqmScores:
    if mqm < 0:
        nonzeromqm.append(mqm)
    else:
        nonzeromqm.append(-0.05)

nonzeromqm = np.log(-(np.array(nonzeromqm)))
print(np.arange(-4, 4, 0.2))
#total.append(Scores40)
#total.append(Scores20)
#total.append(Scores40)

plt.hist(nonzeromqm, 100, histtype ='bar', label=['Testing MQM score dist log on x-axis'])
plt.xticks(np.arange(-4, 4, 0.2))
plt.legend()
plt.show()
