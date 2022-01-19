import csv
import scipy.stats

def calc_spearmanr_corr(x, y):
    return scipy.stats.spearmanr(x, y).correlation

da_scores = []
hter_scores = []
with open("train.ende.df.short.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    next(tsv_file)
    for line in tsv_file:
        da_scores.append(float(line[-2]))

with open("train.hter") as file:
    for line in file:
        hter_scores.append(float(line[:-1]))

print(calc_spearmanr_corr(da_scores, hter_scores))
