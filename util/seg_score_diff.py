import csv
import numpy as np
import matplotlib.pyplot as plt

better_file = open("my_results_better.txt", 'r')
worse_file = open("my_results_worse.txt", 'r')
gt_bfile = open('better-ende-scores.txt', 'r')
gt_wfile = open('worse-ende-scores.txt', 'r')

cor_diff_dist, err_diff_dist = [], []
total = 0
err_eql = 0
cor_gen_dist, err_gen_dist = [], []
cor_thres, incor_thres = 0, 0
big_thres = 0
for better_line, worse_line, gt_bline, gt_wline in zip(better_file, worse_file, gt_bfile, gt_wfile):
    total+=1
    bscore = float(better_line.split("score: ")[1][:-1])
    wscore = float(worse_line.split("score: ")[1][:-1])
    if float(gt_bline[:-1])-float(gt_wline[:-1]) >= 5:
        big_thres+=1
    if bscore > wscore:
        if bscore-wscore <= 1 and float(gt_bline[:-1])-float(gt_wline[:-1]) >= 5:
            cor_thres+=1
        cor_gen_dist.append(bscore-wscore)
        cor_diff_dist.append(float(gt_bline[:-1])-float(gt_wline[:-1]))
    else:
        if float(gt_bline[:-1]) == 0:
            incor_thres+=1
        err_gen_dist.append(bscore-wscore)
        err_diff_dist.append(float(gt_bline[:-1])-float(gt_wline[:-1]))

# print("n1_both_cor", n1_both_cor)
# print("n1_both_err", n1_both_err)
print(total)
print("Predicted equal examples: ", err_eql)

print("cor thres: ", cor_thres)
print("incor thres: ", incor_thres)
print("total big thres: ", big_thres)

plt.title('Generated Cor vs Err Score diff (ende)')
plt.hist([cor_gen_dist, err_gen_dist], 100, histtype ='bar', label=['Generated Cor Dist', 'Generated Err Dist'])
plt.xticks(np.arange(-15, 15, 1))
plt.legend()
plt.show()
