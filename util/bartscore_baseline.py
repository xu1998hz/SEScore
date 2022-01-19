from bart_score import BARTScorer
import torch
import numpy as np

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')

better_lines = open('mqm_better_zhen.txt', 'r').readlines()
worse_lines = open('mqm_worse_zhen.txt', 'r').readlines()
ref_lines = open('eval-zhen-ref.txt', 'r').readlines()
better_bart = []
worse_bart = []

batch_size = 64

for better_line, worse_line, ref_line in zip(batchify(better_lines, batch_size), batchify(worse_lines, batch_size), batchify(ref_lines, batch_size)):
    better_cur_score = bart_scorer.score(better_line, ref_line, batch_size=batch_size)
    better_bart.extend(better_cur_score)
    worse_cur_score = bart_scorer.score(worse_line, ref_line, batch_size=batch_size)
    worse_bart.extend(worse_cur_score)
    print("Finish one batch!")

better_bart = np.array(better_bart)
worse_bart = np.array(worse_bart)

total = better_bart.shape[0]
correct = np.sum(better_bart > worse_bart)
incorrect = total - correct
print("Total: ", total)
print("Correct: ", correct)
print("Kendall Correlation: ", (correct - incorrect)/total)
