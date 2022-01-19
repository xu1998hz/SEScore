
refLines = open('eval-ende-ref.txt', 'r').readlines()
betterLines = open('my_results_better.txt', 'r').readlines()
worseLines = open('my_results_worse.txt', 'r').readlines()

sen_scores_dict = {}
tot_scores_dict = {}

for ref_line, better_line, worse_line in zip(refLines, betterLines, worseLines):
    if ref_line[:-1] not in tot_scores_dict:
        tot_scores_dict[ref_line[:-1]] = 1
    else:
        tot_scores_dict[ref_line[:-1]] += 1
    if float(better_line[:-1].split('score: ')[1]) > float(worse_line[:-1].split('score: ')[1]):
        if ref_line[:-1] not in sen_scores_dict:
            sen_scores_dict[ref_line[:-1]] = 1
        else:
            sen_scores_dict[ref_line[:-1]] += 1

scores_dict = {}
for key, value in sen_scores_dict.items():
    scores_dict[key] = value / tot_scores_dict[key]

max_len = 0
gt_50, gt_count, ls_50, ls_count = 0, 0, 0, 0
for sen, score in scores_dict.items():
    if len(sen.split()) > max_len:
        max_len = len(sen.split())
    if len(sen.split()) > 50:
        gt_50 += score
        gt_count += 1
    else:
        ls_50 += score
        ls_count += 1

print(gt_50/gt_count)
print(ls_50/ls_count)
#print(scores_dict)
