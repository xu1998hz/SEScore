from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import csv
import re

# sanity check on all the score calculation
def sanity_check(mqm_id_scores):
    mqmScoreFile = open('mqm_newstest2020_zhen.avg_seg_scores.tsv', 'r')
    mqmScoreReader = csv.reader(mqmScoreFile, delimiter="\t")
    header = next(mqmScoreReader)
    num_rater = 6
    for row in mqmScoreFile:
        row = row.split()
        msystem, mscore, mid = row[0], float(row[1]), row[2]
        seg_tot_score, count = 0, 0
        for i in range(1, num_rater+1):
            key = mid+'_'+msystem+'_rater'+str(i)
            if key in mqm_id_scores:
                seg_tot_score += mqm_id_scores[key]['score']
                count+=1

        if round(seg_tot_score/count, 4) != round(mscore, 4):
            print(seg_tot_score/count)
            print(mscore)
            print()

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

mqmFile = open('mqm_newstest2020_zhen.tsv', 'r')
mqmReader = csv.reader(mqmFile, delimiter="\t")
header = next(mqmReader)

# estimate score per rater
mqm_id_scores = {}
sen_id_dict = {}
for row in mqmFile:
    row = row.split('\t')
    CLEANR = re.compile('<.*?>')
    msystem, msentence, mid, rater, category, severity = row[0], row[6], row[3], row[4], row[-2], row[-1]
    cleantext = re.sub(CLEANR, '', msentence)
    key = mid+'_'+msystem
    sen_id_dict[cleantext] = key
    if key not in mqm_id_scores:
        mqm_id_scores[key] = {}
        mqm_id_scores[key][rater] = {}
        mqm_id_scores[key][rater]['score'] = 0
        mqm_id_scores[key][rater]['text'] = cleantext
        mqm_id_scores[key][rater]['num_err'] = 0
    else:
        if rater not in mqm_id_scores[key]:
            mqm_id_scores[key][rater] = {}
            mqm_id_scores[key][rater]['score'] = 0
            mqm_id_scores[key][rater]['text'] = cleantext
            mqm_id_scores[key][rater]['num_err'] = 0

    if severity[:-1] == 'Major':
        if category == 'Non-translation!':
            mqm_id_scores[key][rater]['score'] -= 25
            mqm_id_scores[key][rater]['num_err'] += 1
        else:
            mqm_id_scores[key][rater]['score'] -= 5
            mqm_id_scores[key][rater]['num_err'] += 1
    elif severity[:-1] == 'Minor':
        if category == 'Fluency/Punctuation':
            mqm_id_scores[key][rater]['score'] -= 0.1
            mqm_id_scores[key][rater]['num_err'] += 1
        else:
            mqm_id_scores[key][rater]['score'] -= 1
            mqm_id_scores[key][rater]['num_err'] += 1
    else:
        mqm_id_scores[key][rater]['score'] -= 0

ref_lines = open('newstest2020-zhen-ref.en.txt', 'r').readlines()
ref_lines = [line[:-1] for line in ref_lines]
severe_count, minor_count = 0, 0
severe_refs, severe_mts, minor_refs, minor_mts = [], [], [], []
for key, vals in mqm_id_scores.items():
    total_score = 0
    for _, val in vals.items():
        if val['num_err'] == 1 and val['score'] == -5:
            total_score += -5
        elif val['num_err'] == 1 and val['score'] == -1:
            total_score += -1

    if total_score == -15:
        severe_count+=1
        severe_refs.append(ref_lines[int(key.split('_')[0])-1])
        severe_mts.append(val['text'])
    elif total_score == -3:
        minor_count+=1
        minor_refs.append(ref_lines[int(key.split('_')[0])-1])
        minor_mts.append(val['text'])

print("Severe Example Counts: ", severe_count)
print("Minor Example Counts: ", minor_count)

def severity_measure(tokenizer, model, device, m, indicator, thres, mqm_sample, ref_sample, sen_id_dict):
    count = 0
    equal = 0
    cor_locs = {}
    err_locs = {}
    with torch.no_grad():
        for batch_sens, batch_ref in zip(batchify(mqm_sample, 1024), batchify(ref_sample, 1024)):
            # print(batch_sens)
            # print(batch_ref)
            inputs = tokenizer(batch_sens, batch_ref, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
            outputs = model(**inputs).logits
            softmax_result_1 = m(outputs)[:, -1]

            inputs = tokenizer(batch_ref, batch_sens, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
            outputs = model(**inputs).logits
            softmax_result_2 = m(outputs)[:, -1]

            #p_tensor = 2*softmax_result_1*softmax_result_2/(softmax_result_1+softmax_result_2)
            index = 0
            for prob_1, prob_2 in zip(softmax_result_1, softmax_result_2):
                if indicator == 'Severe':
                    if prob_1 <= thres or prob_2 <= thres:
                        count += 1
                    else:
                        if batch_sens[index] == batch_ref[index][:-1]:
                            equal += 1
                        print(prob_1)
                        print(prob_2)
                        print(batch_sens[index])
                        print(batch_ref[index][:-1])
                        print(sen_id_dict[batch_sens[index]])
                        print()
                else:
                    if prob_1 > thres and prob_2 > thres:
                        count += 1
                    else:
                        print(prob_1)
                        print(prob_2)
                        print(batch_sens[index])
                        print(batch_ref[index][:-1])
                        print(sen_id_dict[batch_sens[index]])
                        print()
                        if batch_sens[index] == batch_ref[index][:-1]:
                            equal += 1
                index+=1
    return count, equal

# Define the model repo
model_name = "roberta-large-mnli"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Download pytorch model
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
m = torch.nn.Softmax(dim=1)
minor_cor, equal = severity_measure(tokenizer, model, device, m, 'Minor', 0.9, minor_mts, minor_refs, sen_id_dict)
print(minor_cor/minor_count)
print(equal)
