import numpy as np

better_file = open("my_results_ende_large_norepeat.txt", 'r')
worse_file = open("my_results_ende_large_norepeat_worse.txt", 'r')

better_scores = []
for better_line in better_file:
    score = float(better_line.split("score: ")[1][:-1])
    better_scores.append(score)
better_scores = np.array(better_scores)

worse_scores = []
for worse_line in worse_file:
    score = float(worse_line.split("score: ")[1][:-1])
    worse_scores.append(score)
worse_scores = np.array(worse_scores)

total = better_scores.shape[0]
correct = np.sum(better_scores > worse_scores)
print(correct)
incorrect = total - correct
print((correct - incorrect)/total)
