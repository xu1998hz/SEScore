"""format validation dataset into the format of training csv, compute the score and collect ref and src sentences"""
import re
import csv

def preprocess(text):
    # remove extra space
    text = re.sub(' ', '', text)
    # remove highlights
    CLEANR = re.compile('<.*?>')
    cleantext = re.sub(CLEANR, '', text)
    return cleantext

valiFile = open('wmt-mqm-human-evaluation/ted/zhen/mqm-ted_zhen.tsv', 'r')
mqmReader = csv.reader(valiFile, delimiter="\t")
header = next(mqmReader)

# load in src texts in raw src file
src_lines = open('newstest2021.en-de.src.en', 'r').readlines()
src_ind_dict = {}
for index, src_line in enumerate(src_lines):
    src_ind_dict[re.sub(' ', '', src_line[:-1])] = index
# load in ref texts in raw ref file
ref_lines = open('newstest2021.en-de.ref.ref-B.de', 'r').readlines()

mqm_id_scores = {}
CLEANR = re.compile('<.*?>')
for row in valiFile:
    row = row.split('\t')
    msystem, mid, src_sen, mt_sen, category, severity = row[0], row[3], row[5], row[6], row[-2], row[-1]
    key = mid+'_'+msystem
    if key not in mqm_id_scores:
        mqm_id_scores[key] = {}
        process_src_sen = preprocess(src_sen)
        mqm_id_scores[key]['src'] = process_src_sen
        mqm_id_scores[key]['mt'] = re.sub(CLEANR, '', mt_sen)
        mqm_id_scores[key]['ref'] = ref_lines[src_ind_dict[process_src_sen]][:-1]
        mqm_id_scores[key]['score'] = 0
        mqm_id_scores[key]['num_raters'] = 0

    if severity[:-1] == 'Major':
        if category == 'Non-translation!':
            mqm_id_scores[key]['score'] -= 25
            mqm_id_scores[key]['num_raters'] += 1
        else:
            mqm_id_scores[key]['score'] -= 5
            mqm_id_scores[key]['num_raters'] += 1
    elif severity[:-1] == 'Minor':
        if category == 'Fluency/Punctuation':
            mqm_id_scores[key]['score'] -= 0.1
            mqm_id_scores[key]['num_raters'] += 1
        else:
            mqm_id_scores[key]['score'] -= 1
            mqm_id_scores[key]['num_raters'] += 1
    else:
        mqm_id_scores[key]['score'] -= 0
        mqm_id_scores[key]['num_raters'] += 1

csvfile = open('zhen_vali_TED.csv', 'w')
csvwriter = csv.writer(csvfile)
fields = ['src', 'mt', 'ref', 'score']
csvwriter.writerow(fields)
less_n = 0
for key, value in mqm_id_scores.items():
    if mqm_id_scores[key]['num_raters'] < 3:
        less_n+=1
        print("raters less than 3")
    csvwriter.writerow([mqm_id_scores[key]['src'], mqm_id_scores[key]['mt'], mqm_id_scores[key]['ref'], float(mqm_id_scores[key]['score'])/mqm_id_scores[key]['num_raters']])

print(less_n)
print(len(mqm_id_scores))
print("validation file is generated!")
