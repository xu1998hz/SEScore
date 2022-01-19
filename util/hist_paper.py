import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def score_ls_gen(file_name):
    mqmScoreFile = open(file_name, 'r')
    mqmScoreReader = csv.reader(mqmScoreFile, delimiter="\t")
    header = next(mqmScoreReader)

    mqmScores = []
    n5_count = 0
    total = 0
    for row in mqmScoreFile:
        row = row.split()
        msystem, mscore, mid = row[0], row[1], row[2]
        if float(mscore) <= -5:
            n5_count += 1
        total += 1
        mqmScores.append(float(mscore))

    return mqmScores, float(n5_count)/total, total


de_mqmScores, percent_de, de_total = score_ls_gen('mqm_newstest2020_ende.avg_seg_scores.tsv')
en_mqmScores, percent_en, en_total = score_ls_gen('mqm_newstest2020_zhen.avg_seg_scores.tsv')

print(f"De: {percent_de}")
print(f"En: {percent_en}")

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

plt.hist(de_mqmScores, 100, histtype ='bar', label=['En-De MQM Score Distribution'])

plt.gca().set_yticklabels(['{:.0f}%'.format(float(x)/de_total*100) for x in plt.gca().get_yticks()])
plt.xticks(np.arange(0, -26, -5))
plt.legend()
plt.savefig('ende_mqm.pdf')
