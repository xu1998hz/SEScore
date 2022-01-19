import csv

csvfile = open('mqm_newstest2020_zhen.avg_seg_scores.tsv', 'r')
csvreader = csv.reader(csvfile, delimiter="\t")
header = next(csvreader)

print(csvreader)
