import os
import numpy as np
from prism import Prism

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

prism = Prism(model_dir=os.environ['MODEL_DIR'], lang='en')

better_lines = open('mqm_better_zhen.txt', 'r').readlines()
worse_lines = open('mqm_worse_zhen.txt', 'r').readlines()
ref_lines = open('eval-zhen-ref.txt', 'r').readlines()
better_prism = []
worse_prism = []

for better_line, worse_line, ref_line in zip(batchify(better_lines, 2048), batchify(worse_lines, 2048), batchify(ref_lines, 2048)):
    better_cur_score = prism.score(cand=better_line, ref=ref_line, segment_scores=True)
    better_prism.extend(better_cur_score)
    worse_cur_score = prism.score(cand=worse_line, ref=ref_line, segment_scores=True)
    worse_prism.extend(worse_cur_score)
    print("Finish one batch!")

better_prism = np.array(better_prism)
worse_prism = np.array(worse_prism)

total = better_prism.shape[0]
correct = np.sum(better_prism > worse_prism)
incorrect = total - correct
print("Total: ", total)
print("Correct: ", correct)
print("Kendall Correlation: ", (correct - incorrect)/total)
