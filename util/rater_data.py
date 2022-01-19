"""extract one rater's data per sentence to train the model and evaluate on the testing set"""
"""format validation dataset into the format of training csv, compute the score and collect ref and src sentences"""
import re
import matplotlib.pyplot as plt
import csv
import numpy as np

def preprocess(text):
    # remove extra space
    text = re.sub(' ', '', text)
    # remove highlights
    CLEANR = re.compile('<.*?>')
    cleantext = re.sub(CLEANR, '', text)
    return cleantext

def print_score_count(scores_ls):
    range_0 = 0
    range_1 = 0
    range_2 = 0
    range_3 = 0
    range_4 = 0
    range_5 = 0
    range_6 = 0
    range_7 = 0
    range_8 = 0
    range_9 = 0
    range_10 = 0
    range_0_1 = 0
    range_1_5 = 0
    range_5_10 = 0
    range_10_15 = 0
    range_15_20 = 0
    range_20_25 = 0
    count = 0
    for score in scores_ls:
        count += 1
        if score > -1 and score <= 0:
            if score == 0:
                range_0 += 1
            range_0_1 += 1
        elif score > -5 and score <= -1:
            if score == -1:
                range_1 += 1
            elif score == -2:
                range_2 += 1
            elif score == -3:
                range_3 += 1
            elif score == -4:
                range_4 += 1

            range_1_5 += 1

        elif score > -10 and score <= -5:
            if score == -5:
                range_5 += 1
            if score == -6:
                range_6 += 1
            if score == -7:
                range_7 += 1
            if score == -8:
                range_8 += 1
            if score == -9:
                range_9 += 1

            range_5_10 += 1
        elif score > -15 and score <= -10:
            if score == -10:
                range_10 += 1
            range_10_15 += 1
        elif score > -20 and score <= -15:
            range_15_20 += 1
        else:
            range_20_25 += 1

    print("For current rater: ")
    print("0: ", range_0/count)
    print("-1: ", range_1/count)
    print(range_1)
    print("-2: ", range_2/count)
    print("-3: ", range_3/count)
    print("-4: ", range_4/count)
    print("-5: ", range_5/count)
    print(range_5)
    print("-6: ", range_6/count)
    print("-7: ", range_7/count)
    print("-8: ", range_8/count)
    print("-9: ", range_9/count)
    print("-10: ", range_10/count)
    print("range 0 ~ -1: ", range_0_1/count)
    print("range -1 ~ -5: ", range_1_5/count)
    print("range -5 ~ -10: ", range_5_10/count)
    print("range -10 ~ -15: ", range_10_15/count)
    print("range -15 ~ -20: ", range_15_20/count)
    print("range -20 ~ -25: ", range_20_25/count)
    print("-------------------------------------------------")

valiFile = open('mqm_newstest2020_zhen.tsv', 'r')
mqmReader = csv.reader(valiFile, delimiter="\t")
header = next(mqmReader)

# load in src texts in raw src file
src_lines = open('newstest2020-zhen-src.zh.txt', 'r').readlines()
src_ind_dict = {}
for index, src_line in enumerate(src_lines):
    src_ind_dict[re.sub(' ', '', src_line[:-1])] = index
# load in ref texts in raw ref file
ref_lines = open('newstest2020-zhen-ref.en.txt', 'r').readlines()

mqm_id_scores = {}
CLEANR = re.compile('<.*?>')
for row in valiFile:
    row = row.split('\t')
    msystem, mid, rater, src_sen, mt_sen, category, severity = row[0], row[3], row[4], row[5], row[6], row[-2], row[-1]
    key = mid+'_'+msystem
    process_src_sen = preprocess(src_sen)
    # initialize the sys and seg id
    if key not in mqm_id_scores:
        mqm_id_scores[key] = {}
    # initialize the rater
    if rater not in mqm_id_scores[key]:
        mqm_id_scores[key][rater] = {}
        mqm_id_scores[key][rater]['src'] = process_src_sen
        mqm_id_scores[key][rater]['mt'] = re.sub(CLEANR, '', mt_sen)
        mqm_id_scores[key][rater]['ref'] = ref_lines[src_ind_dict[process_src_sen]][:-1]
        mqm_id_scores[key][rater]['score'] = 0

    if severity[:-1] == 'Major':
        if category == 'Non-translation!':
            mqm_id_scores[key][rater]['score'] -= 25
        else:
            mqm_id_scores[key][rater]['score'] -= 5
    elif severity[:-1] == 'Minor':
        if category == 'Fluency/Punctuation':
            mqm_id_scores[key][rater]['score'] -= 0.1
        else:
            mqm_id_scores[key][rater]['score'] -= 1
    else:
        mqm_id_scores[key][rater]['score'] -= 0

csvfile = open('zhen_vali_rater.csv', 'w')
csvwriter = csv.writer(csvfile)
fields = ['src', 'mt', 'ref', 'score']
score_ls_1, score_ls_2, score_ls_3 = [], [], []
csvwriter.writerow(fields)
for key, value in mqm_id_scores.items():
    cur_rater_1, cur_rater_2, cur_rater_3 = tuple(value.keys())
    #if float(mqm_id_scores[key][cur_rater_1]['score']) < 0:
    score_ls_1.append(float(mqm_id_scores[key][cur_rater_1]['score']))
    #if float(mqm_id_scores[key][cur_rater_2]['score']) < 0:
    score_ls_2.append(float(mqm_id_scores[key][cur_rater_2]['score']))
    #if float(mqm_id_scores[key][cur_rater_3]['score']) < 0:
    score_ls_3.append(float(mqm_id_scores[key][cur_rater_3]['score']))
    csvwriter.writerow([mqm_id_scores[key][cur_rater_1]['src'], mqm_id_scores[key][cur_rater_1]['mt'], mqm_id_scores[key][cur_rater_1]['ref'], float(mqm_id_scores[key][cur_rater_1]['score'])])

plt.title('Zh-En single rater distribution (scores < 0)')
print_score_count(score_ls_1)
print_score_count(score_ls_2)
print_score_count(score_ls_3)
# plt.hist([score_ls_1, score_ls_2, score_ls_3], 100, histtype ='bar', label=['rater1', 'rater2', 'rater3'])
# plt.xticks(np.arange(-25, 0, 1))
# plt.legend()
# plt.show()
# print("validation file is generated!")
