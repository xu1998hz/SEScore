import nltk
import click
import numpy as np

better_lines = open('data/overall_better.txt', 'r').readlines()
worse_lines = open('data/overall_worse.txt', 'r').readlines()
ref_lines = open('data/gt_overall.txt', 'r').readlines()

better = []
worse = []

for better_line, worse_line, ref_line in zip(better_lines, worse_lines, ref_lines):
    better_score = nltk.translate.bleu_score.sentence_bleu([ref_line], better_line)
    better.append(better_score)
    worse_score = nltk.translate.bleu_score.sentence_bleu([ref_line], worse_line)
    worse.append(worse_score)

better = np.array(better)
worse = np.array(worse)

total = better.shape[0]
correct = np.sum(better > worse)
incorrect = total - correct
print("Total: ", total)
print("Correct: ", correct)
print("Kendall Correlation: ", (correct - incorrect)/total)
