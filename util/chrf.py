import nltk
import click
import numpy as np

better_lines = open('mqm_better_zhen.txt', 'r').readlines()
worse_lines = open('mqm_worse_zhen.txt', 'r').readlines()
ref_lines = open('eval-zhen-ref.txt', 'r').readlines()

better = []
worse = []

for better_line, worse_line, ref_line in zip(better_lines, worse_lines, ref_lines):
    better_score = nltk.translate.chrf_score.sentence_chrf(ref_line.split(), better_line.split())
    better.append(better_score)
    worse_score = nltk.translate.chrf_score.sentence_chrf(ref_line.split(), worse_line.split())
    worse.append(worse_score)

better = np.array(better)
worse = np.array(worse)

total = better.shape[0]
correct = np.sum(better > worse)
incorrect = total - correct
print("Total: ", total)
print("Correct: ", correct)
print("Kendall Correlation: ", (correct - incorrect)/total)
