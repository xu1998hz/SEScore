import pyter
import click
import numpy as np

better_lines = open('overall_better.txt', 'r').readlines()
worse_lines = open('overall_worse.txt', 'r').readlines()
ref_lines = open('gt_overall.txt', 'r').readlines()

better = []
worse = []
count = 0

for better_line, worse_line, ref_line in zip(better_lines, worse_lines, ref_lines):
    better_score = pyter.ter(better_line.split(), ref_line.split())
    better.append(better_score)
    worse_score = pyter.ter(worse_line.split(), ref_line.split())
    worse.append(worse_score)
    count += 1
    print(count)

better = np.array(better)
worse = np.array(worse)

total = better.shape[0]
correct = np.sum(better > worse)
incorrect = total - correct
print("Total: ", total)
print("Correct: ", correct)
print("Kendall Correlation: ", (correct - incorrect)/total)
