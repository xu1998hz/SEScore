from bleurt import score

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

scorer = score.BleurtScorer()
df = pd.read_csv('zhen-comet.csv', index_col=False)
df = df[["ref", "mt", "score"]]
ref_ls = df["ref"].astype(str).tolist()
mt_ls = df["mt"].astype(str).tolist()
score_ls = df["score"].astype(float).tolist()


for better_line, worse_line, ref_line in zip(batchify(better_lines, 128), batchify(worse_lines, 128), batchify(ref_lines, 128)):
    better_cur_score = scorer.score(candidates=better_line, references=ref_line)
    better_bert.extend(better_cur_score)
    worse_cur_score = scorer.score(candidates=worse_line, references=ref_line)
    worse_bert.extend(worse_cur_score)

better_bert = np.array(better_bert)
worse_bert = np.array(worse_bert)

total = better_bert.shape[0]
correct = np.sum(better_bert > worse_bert)
incorrect = total - correct
print("Total: ", total)
print("Correct: ", correct)
print("Kendall Correlation: ", (correct - incorrect)/total)
