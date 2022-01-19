from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import csv
import re
import json

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

mqmScoreFile = open('mqm_newstest2020_zhen.avg_seg_scores.tsv', 'r')
mqmScoreReader = csv.reader(mqmScoreFile, delimiter="\t")
header = next(mqmScoreReader)

mqmFile = open('mqm_newstest2020_zhen.tsv', 'r')
mqmReader = csv.reader(mqmFile, delimiter="\t")
header = next(mqmReader)

mqm_dict = {}
sen_mqm_dict = {}
for row in mqmFile:
    row = row.split('\t')
    CLEANR = re.compile('<.*?>')
    msystem, msentence, mid = row[0], row[6], row[3]
    cleantext = re.sub(CLEANR, '', msentence)
    mqm_dict[mid+'_'+msystem] = cleantext
    sen_mqm_dict[cleantext] = mid+'_'+msystem
print("mqm sentences are loaded!")

mqmSevere, mqmMinor = [], []
minor_count, major_count = 0, 0
ref_lines = open("newstest2020-zhen-ref.en.txt", 'r').readlines()
refSevere, refMinor = [], []
for row in mqmScoreFile:
    row = row.split()
    msystem, mscore, mid = row[0], float(row[1]), row[2]
    if mscore == -1:
        minor_count += 1
        mqmMinor.append(mqm_dict[mid+'_'+msystem])
        refMinor.append(ref_lines[int(mid)-1])
    elif mscore == -5:
        major_count += 1
        mqmSevere.append(mqm_dict[mid+'_'+msystem])
        refSevere.append(ref_lines[int(mid)-1])

print("Minor Count: ", minor_count)
print("Major Count: ", major_count)
print("Ref sentences are loaded!")

# Define the model repo
model_name = "roberta-large-mnli"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Download pytorch model
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
m = torch.nn.Softmax(dim=1)

def severity_measure(tokenizer, model, device, m, indicator, thres, sen_mqm_dict, mqm_sample, ref_sample):
    count = 0
    cor_locs = {}
    err_locs = {}
    with torch.no_grad():
        for batch_sens, batch_ref in zip(batchify(mqm_sample, 64), batchify(ref_sample, 64)):
            # print(batch_sens)
            # print(batch_ref)
            inputs = tokenizer(batch_sens, batch_ref, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
            outputs = model(**inputs).logits
            softmax_result_1 = m(outputs)[:, -1]

            inputs = tokenizer(batch_ref, batch_sens, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
            outputs = model(**inputs).logits
            softmax_result_2 = m(outputs)[:, -1]

            p_tensor = 2*softmax_result_1*softmax_result_2/(softmax_result_1+softmax_result_2)

            for index, p in enumerate(p_tensor):
                if indicator == 'Severe':
                    if p <= thres:
                        count += 1
                        cor_locs[sen_mqm_dict[batch_sens[index]]] = [p.item(), 'Severe']
                    else:
                        err_locs[sen_mqm_dict[batch_sens[index]]] = [p.item(), 'Severe']
                else:
                    if p > thres:
                        count += 1
                        cor_locs[sen_mqm_dict[batch_sens[index]]] = [p.item(), 'Minor']
                    else:
                        err_locs[sen_mqm_dict[batch_sens[index]]] = [p.item(), 'Minor']
    return cor_locs, err_locs, count

s_cor_dict, s_err_dict, s_count = severity_measure(tokenizer, model, device, m, 'Severe', 0.8, sen_mqm_dict, mqmSevere, refSevere)
print(s_count)
print(s_count/major_count)
m_cor_dict, m_err_dict, m_count = severity_measure(tokenizer, model, device, m, 'Minor', 0.8, sen_mqm_dict, mqmMinor, refMinor)
print(m_count)
print(m_count/minor_count)
s_cor_saveFile = open('s_cor_dist.json', 'w')
m_cor_saveFile = open('m_cor_dist.json', 'w')
s_err_saveFile = open('s_err_dist.json', 'w')
m_err_saveFile = open('m_err_dist.json', 'w')
json.dump(s_cor_dict, s_cor_saveFile)
json.dump(m_cor_dict, m_cor_saveFile)
json.dump(s_err_dict, s_err_saveFile)
json.dump(m_err_dict, m_err_saveFile)
