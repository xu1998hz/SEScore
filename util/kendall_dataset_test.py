"""format validation dataset into the format of training csv, compute the score and collect ref and src sentences"""
import re
import csv
from itertools import combinations

def preprocess(text):
    # remove extra space
    text = re.sub(' ', '', text)
    # remove highlights
    CLEANR = re.compile('<.*?>')
    cleantext = re.sub(CLEANR, '', text)
    return cleantext

dir = 'ende'
src_lang = 'en'
label = '2021'

valiFile = open('wmt-mqm-human-evaluation/newstest2021/ende/mqm-newstest2021_ende.tsv', 'r')
mqmReader = csv.reader(valiFile, delimiter="\t")
header = next(mqmReader)

# load in src texts in raw src file
src_lines = open('newstest2021.en-de.src.en', 'r').readlines()
src_ind_dict = {}
for index, src_line in enumerate(src_lines):
    if src_lang == 'zh':
        src_ind_dict[re.sub(' ', '', src_line[:-1])] = index
    else:
        src_ind_dict[src_line[:-1]] = index
# load in ref texts in raw ref file
ref_lines = open('newstest2021.en-de.ref.ref-B.de', 'r').readlines()
#print(src_ind_dict)

assert(len(src_lines) == len(ref_lines))

mqm_id_scores = {}
CLEANR = re.compile('<.*?>')
system_name = set()
# calculate segment score and store all system names
count = 0
for row in valiFile:
    row = row.split('\t')
    msystem, mid, src_sen, mt_sen, category, severity = row[0], int(row[3]), row[5], row[6], row[-2], row[-1]
    system_name.add(msystem)
    if mid not in mqm_id_scores:
        mqm_id_scores[mid] = {}

        if src_lang == 'zh':
            process_src_sen = preprocess(src_sen)
        else:
            process_src_sen = src_sen

        mqm_id_scores[mid]['src_wenda'] = process_src_sen
        # print()
        #
        if process_src_sen not in src_ind_dict:
            mqm_id_scores[mid]['ref_wenda'] = 'null'
        else:
            mqm_id_scores[mid]['ref_wenda'] = ref_lines[src_ind_dict[process_src_sen]][:-1]

    if msystem not in mqm_id_scores[mid]:
        mqm_id_scores[mid][msystem] = {}
        mqm_id_scores[mid][msystem]['mt'] = re.sub(CLEANR, '', mt_sen)
        mqm_id_scores[mid][msystem]['score'] = 0
        mqm_id_scores[mid][msystem]['num_raters'] = 0

    if severity[:-1] == 'Major':
        if category == 'Non-translation!':

            mqm_id_scores[mid][msystem]['score'] -= 25
            mqm_id_scores[mid][msystem]['num_raters'] += 1
        else:
            # print('-------------------------------------------')
            # print(mqm_id_scores[mid])
            # print()
            # print(msystem)
            mqm_id_scores[mid][msystem]['score'] -= 5
            mqm_id_scores[mid][msystem]['num_raters'] += 1
    elif severity[:-1] == 'Minor':
        if category == 'Fluency/Punctuation':
            mqm_id_scores[mid][msystem]['score'] -= 0.1
            mqm_id_scores[mid][msystem]['num_raters'] += 1
        else:
            mqm_id_scores[mid][msystem]['score'] -= 1
            mqm_id_scores[mid][msystem]['num_raters'] += 1
    else:
        mqm_id_scores[mid][msystem]['score'] -= 0
        mqm_id_scores[mid][msystem]['num_raters'] += 1

num_seg = len(ref_lines)
sys_comb = list(combinations(system_name,2))

betterFile = open(f'mqm_better_{dir}_{label}.txt', 'w')
worseFile = open(f'mqm_worse_{dir}_{label}.txt', 'w')
outSrcFile = open(f'eval-{dir}-src_{label}.txt', 'w')
outRefFile = open(f'eval-{dir}-ref_{label}.txt', 'w')

for i in mqm_id_scores:
    for sys_pair in sys_comb:
        sysA, sysB = sys_pair
        if sysA in mqm_id_scores[i] and sysB in mqm_id_scores[i]:
            if mqm_id_scores[i]['ref_wenda'] != 'null':
                if mqm_id_scores[i][sysA]['score']/mqm_id_scores[i][sysA]['num_raters'] > mqm_id_scores[i][sysB]['score']/mqm_id_scores[i][sysB]['num_raters']:
                    outSrcFile.write(mqm_id_scores[i]['src_wenda']+'\n')
                    outRefFile.write(mqm_id_scores[i]['ref_wenda']+'\n')
                    betterFile.write(mqm_id_scores[i][sysA]['mt']+'\n')
                    worseFile.write(mqm_id_scores[i][sysB]['mt']+'\n')
                elif mqm_id_scores[i][sysA]['score']/mqm_id_scores[i][sysA]['num_raters'] < mqm_id_scores[i][sysB]['score']/mqm_id_scores[i][sysB]['num_raters']:
                    outSrcFile.write(mqm_id_scores[i]['src_wenda']+'\n')
                    outRefFile.write(mqm_id_scores[i]['ref_wenda']+'\n')
                    betterFile.write(mqm_id_scores[i][sysB]['mt']+'\n')
                    worseFile.write(mqm_id_scores[i][sysA]['mt']+'\n')

print("Four files generated!")
