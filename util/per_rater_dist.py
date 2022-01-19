import tsv
import re

mqmFile = open('mqm_newstest2020_zhen.tsv', 'r')
mqmReader = csv.reader(mqmFile, delimiter="\t")
header = next(mqmReader)

mqm_id_scores = {}
# intialize all 6 raters
for i in range(1, 7):
    mqm_id_scores[f'rater{i}'] = {}
# for each rater record all their given scores based on the segment per system
for row in mqmFile:
    row = row.split('\t')
    CLEANR = re.compile('<.*?>')
    msystem, mid, rater, category, severity = row[0], row[3], row[4], row[-2], row[-1]

    if key not in mqm_id_scores[rater]:
        mqm_id_scores[rater][key] = {}
        mqm_id_scores[rater][key]['score'] = 0
        mqm_id_scores[key][rater]['num_err'] = 0

    if severity[:-1] == 'Major':
        if category == 'Non-translation!':
            mqm_id_scores[rater][key]['score'] -= 25
            mqm_id_scores[rater][key]['num_err'] += 1
        else:
            mqm_id_scores[rater][key]['score'] -= 5
            mqm_id_scores[rater][key]['num_err'] += 1
    elif severity[:-1] == 'Minor':
        if category == 'Fluency/Punctuation':
            mqm_id_scores[rater][key]['score'] -= 0.1
            mqm_id_scores[rater][key]['num_err'] += 1
        else:
            mqm_id_scores[rater][key]['score'] -= 1
            mqm_id_scores[rater][key]['num_err'] += 1
    else:
        mqm_id_scores[rater][key]['score'] -= 0
