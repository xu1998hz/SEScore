import bert_score
import numpy as np

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

scorer = bert_score.scorer.BERTScorer(model_type='bert-base-multilingual-cased')
better_lines = open('data/better_grammar.txt', 'r').readlines()
worse_lines = open('data/worse_grammar.txt', 'r').readlines()
ref_lines = open('data/gt_grammar.txt', 'r').readlines()
better_bert = []
worse_bert = []

for better_line, worse_line, ref_line in zip(batchify(better_lines, 64), batchify(worse_lines, 64), batchify(ref_lines, 64)):
    _, _, better_cur_score = scorer.score(better_line, ref_line, verbose=False)
    better_bert.extend(better_cur_score)
    _, _, worse_cur_score = scorer.score(worse_line, ref_line, verbose=False)
    worse_bert.extend(worse_cur_score)

better_bert = np.array(better_bert)
worse_bert = np.array(worse_bert)

total = better_bert.shape[0]
correct = np.sum(better_bert > worse_bert)
incorrect = total - correct
print("Total: ", total)
print("Correct: ", correct)
print("Kendall Correlation: ", (correct - incorrect)/total)
