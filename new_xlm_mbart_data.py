import random
import click
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn
import csv
import numpy as np
import re
from scipy.stats import poisson
import time
from transformers import MBartForConditionalGeneration, MBartTokenizer
import glob

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

def noise_sanity_check(cand_arr, num_noises, del_noise_lam, mask_noise_lam):
    # decide noise type upon function called, only sentences have one noise and step 1 can have MBart noises
    if num_noises == 1:
        noise_type = random.choices([1, 2, 3, 4, 5, 6], weights=(4, 4, 1, 1, 0, 1), k=1)[0]
    else:
        noise_type = random.choices([3, 4, 5, 6], weights=(1, 1, 0, 1), k=1)[0]

    if noise_type == 1 or noise_type == 2:
        start_index = random.choices(range(cand_arr['mbart'].shape[0]), k=1)[0]
    else:
        start_index = random.choices(range(cand_arr['xlm'].shape[0]), k=1)[0]
    # this is the MBart addition noise
    if noise_type == 1:
        # check if noise position and span length fits current noise context
        if cand_arr['mbart'][start_index] > 0:
            return noise_type, start_index, 0
    # this is the MBart replace noise which can replace the span of noises
    elif noise_type == 2:
        num_replace = random.choices([1, 2, 3, 4, 5, 6], k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr['mbart'][start_index] >= num_replace and cand_arr['mbart'][start_index] != 0:
            return noise_type, start_index, num_replace
    # this is the XLM addition noise
    elif noise_type == 3:
        if cand_arr['xlm'][start_index] > 0 and cand_arr['xlm'][start_index] != 0:
            return noise_type, start_index, 0
    # this is the XLM replace noise
    elif noise_type == 4:
        if cand_arr['xlm'][start_index] > 0 and cand_arr['xlm'][start_index] != 0:
            return noise_type, start_index, 1
    # this is the swap noise
    elif noise_type == 5:
        if cand_arr['xlm'].shape[0] > 2:
            # within range 4 choose the second index
            start_index = random.choices(range(cand_arr['xlm'].shape[0]-4), k=1)[0] # indices = sorted(random.sample(range(cand_arr['xlm'].shape[0]), 2))
            end_index = start_index + random.choices([1,2,3,4], k=1)[0] # start_index, end_index = indices[0], indices[1]
            if cand_arr['xlm'][start_index] > 0 and cand_arr['xlm'][end_index] > 0:
                return noise_type, start_index, end_index
    # this is the delete noise
    else:
        num_deletes = random.choices([1, 2, 3, 4], k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr['xlm'][start_index] >= num_deletes and cand_arr['xlm'][start_index] != 0:
            return noise_type, start_index, num_deletes
    return -1, -1, -1

"""return planned noise combinations for each sentence with num_var variances"""
def noise_planner(num_var, num_texts, lam):
    sen_noise_dict = {}
    max_step = 0
    for sen_index in range(num_texts):
        for noise_index in range(num_var):
            # Random selection of number of noises
            num_noises = random.choices([1, 2, 3, 4, 5], weights=poisson.pmf(np.arange(1, 6, 1), mu=lam, loc=1), k=1)[0]
            sen_noise_dict[str(sen_index)+'_'+str(noise_index)] = num_noises
            if num_noises > max_step:
                max_step = num_noises
    return sen_noise_dict, max_step # return in dict: key->segID_noiseID, value->num of noises (A list of noise types)

"""seq list dict: key is step index, value is a dict of sentences: key is the segID_noiseID. value is the modifed sentence
    dict: key->segID_noiseID, value->[original sentence, noise1, noise2, ... noisek]"""
def noise_schedule(id_sen_dict, step, sen_noise_dict, cand_dict_arr, del_noise_lam, mask_noise_lam):
    # all the necessary information to construct all 6 noise types
    mbart_add_seg_id_ls, mbart_add_start_ls = [], []
    mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls = [], [], []
    xlm_add_seg_id_ls, xlm_add_start_ls = [], []
    xlm_replace_seg_id_ls, xlm_replace_start_ls = [], []
    swap_seg_id_ls, swap_start_ls, swap_end_ls = [], [], []
    del_seg_id_ls, del_start_ls, del_len_ls = [], [], []
    step_noise_dict = {}
    # check if the given noise type and condition is valid and return the valid noise and condition
    for id, num_noises in sen_noise_dict.items():
        # check if the segment has the valid number of noise for current step
        if step <= num_noises:
            noise_type, start_index, num_ops = noise_sanity_check(cand_dict_arr[id], num_noises, del_noise_lam, mask_noise_lam)
            # only if random selected error type and error number is valid
            if noise_type != -1:
                # type1: MBart Addition noise
                if noise_type == 1: # store mbart add start index and corresponding seg id
                    mbart_add_seg_id_ls.append(id)
                    mbart_add_start_ls.append(start_index)
                    step_noise_dict[id] = ['MBart Addition', start_index]
                # type2: MBart replace noise
                elif noise_type == 2:
                    mbart_replace_seg_id_ls.append(id)
                    mbart_replace_start_ls.append(start_index)
                    mbart_replace_len_ls.append(num_ops)
                    step_noise_dict[id] = ['MBart Replace', start_index, num_ops]
                # type3: XLM-Roberta Addition
                elif noise_type == 3:
                    xlm_add_seg_id_ls.append(id)
                    xlm_add_start_ls.append(start_index)
                    step_noise_dict[id] = ['XLM Addition', start_index]
                # type4: XLM-Roberta Replace
                elif noise_type == 4:
                    xlm_replace_seg_id_ls.append(id)
                    xlm_replace_start_ls.append(start_index)
                    step_noise_dict[id] = ['XLM Replace', start_index, 1]
                # type6: Swap noise
                elif noise_type == 5:
                    swap_seg_id_ls.append(id)
                    swap_start_ls.append(start_index)
                    swap_end_ls.append(num_ops)
                    step_noise_dict[id] = ['Switch', start_index, num_ops]
                else: # Accuracy/Omission, Fluency/Grammar
                    del_seg_id_ls.append(id)
                    del_start_ls.append(start_index)
                    del_len_ls.append(num_ops)
                    step_noise_dict[id] = ['Delete', start_index, num_ops]

    # seg_id_ls: a list contains all the seg_noise ids
    return mbart_add_seg_id_ls, mbart_add_start_ls, mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls, xlm_add_seg_id_ls, xlm_add_start_ls, \
    xlm_replace_seg_id_ls, xlm_replace_start_ls, swap_seg_id_ls, swap_start_ls, swap_end_ls, del_seg_id_ls, del_start_ls, del_len_ls, step_noise_dict

"""add operation to update the candidate dict"""
def add_update_cand_dict(cand_dict_arr, add_seg_id_ls, add_start_ls):
    for add_seg_id, add_start in zip(add_seg_id_ls, add_start_ls):
        new_cand_ls = []
        # update all the values on the left
        for index, _ in enumerate(cand_dict_arr[add_seg_id]['xlm'][:add_start+1]):
            new_cand_ls.append(min(add_start-index, cand_dict_arr[add_seg_id]['xlm'][index]))
        new_cand_ls.extend([0])
        new_cand_ls.extend(list(cand_dict_arr[add_seg_id]['xlm'][add_start+1:]))
        cand_dict_arr[add_seg_id]['xlm'] = np.array(new_cand_ls)
    return cand_dict_arr

"""replace operation to update the candidate dict"""
def replace_update_cand_dict(cand_dict_arr, replace_seg_id_ls, replace_start_ls):
    for replace_seg_id, replace_start in zip(replace_seg_id_ls, replace_start_ls):
        new_cand_ls = []
        # update all the values on the left
        for index, _ in enumerate(cand_dict_arr[replace_seg_id]['xlm'][:replace_start+1]):
            new_cand_ls.append(min(replace_start-index, cand_dict_arr[replace_seg_id]['xlm'][index]))
        new_cand_ls.extend([0])
        new_cand_ls.extend(list(cand_dict_arr[replace_seg_id]['xlm'][replace_start+2:]))
        cand_dict_arr[replace_seg_id]['xlm'] = np.array(new_cand_ls)
    return cand_dict_arr

4 3 2 1 0
1 0 0 2 1 0

"""delete operation to update the candidate dict"""
def delete_update_cand_dict(cand_dict_arr, del_seg_id_ls, del_start_ls, del_len_ls):
    for del_seg_id, del_start, del_len in zip(del_seg_id_ls, del_start_ls, del_len_ls):
        new_cand_ls = []
        # update all the values on the left
        for index in range(len(cand_dict_arr[del_seg_id]['xlm'][:del_start+1])):
            new_cand_ls.append(min(del_start-index, cand_dict_arr[del_seg_id]['xlm'][index]))
        new_cand_ls.extend(list(cand_dict_arr[del_seg_id]['xlm'][del_start+1+del_len:]))
        cand_dict_arr[del_seg_id]['xlm'] = np.array(new_cand_ls)
    return cand_dict_arr

def prev_ids_sens_extract(id_sen_dict, new_seg_ids):
    prev_sen_ls = []
    for id in new_seg_ids:
        prev_sen_ls.append(id_sen_dict[id]['text'][-1])
    return prev_sen_ls

def swap_update_cand_dict(cand_dict_arr, swap_seg_id_ls, swap_start_ls, swap_end_ls):
    for swap_seg_id, swap_start, swap_end in zip(swap_seg_id_ls, swap_start_ls, swap_end_ls):
        new_cand_ls = []
        for index in range(len(cand_dict_arr[swap_seg_id]['xlm'][:swap_start+1])):
            new_cand_ls.append(min(swap_start-index, cand_dict_arr[swap_seg_id]['xlm'][index]))
        new_cand_ls += [0]
        for index in range(len(cand_dict_arr[swap_seg_id]['xlm'][swap_start+2:swap_end+1])):
            new_cand_ls.append(min(swap_end-swap_start-index, cand_dict_arr[swap_seg_id]['xlm'][swap_start+2+index]))
        new_cand_ls += [0] + list(cand_dict_arr[swap_seg_id]['xlm'][swap_end+2:])
        cand_dict_arr[swap_seg_id]['xlm'] = np.array(new_cand_ls)
    return cand_dict_arr

def severity_measure(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device):
    with torch.no_grad():
        inputs_1 = mnli_tokenizer(prev_batch, cur_batch, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
        output_1 = mnli_model(**inputs_1).logits # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_1 = m(output_1)[:, -1]

        inputs_2 = mnli_tokenizer(cur_batch, prev_batch, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
        output_2 = mnli_model(**inputs_2).logits # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_2 = m(output_2)[:, -1]

    # Use the harmonic mean for threshold, threshold = 0.9
    # p_not_severe = softmax_result_1/(1-softmax_result_1) * softmax_result_2/(1-softmax_result_2)
    scores = []
    for prob_1, prob_2 in zip(softmax_result_1, softmax_result_2):
        if prob_1 > 0.9 and prob_2 > 0.9:
            scores.append(-1)
        else:
            scores.append(-5)
    return scores

def data_construct(text, noise_type, tokenizer, start_index, num_ops):
    start_index += 1 # to incorporate beginning token
    if noise_type == 1:
        sen = '</s> ' + text
        tok_text = tokenizer(sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)['input_ids']
        input_ids = torch.cat((tok_text[0][:start_index+1], torch.LongTensor([tokenizer.mask_token_id]), tok_text[0][start_index+1:]), dim=0)
    elif noise_type == 2:
        sen = '</s> ' + text
        tok_text = tokenizer(sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)['input_ids']
        input_ids = torch.cat((tok_text[0][:start_index+1], torch.LongTensor([tokenizer.mask_token_id]), tok_text[0][start_index+1+num_ops:]), dim=0)
    elif noise_type == 3:
        tok_text = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)['input_ids']
        input_ids = torch.cat((tok_text[0][:start_index+1], torch.LongTensor([tokenizer.mask_token_id]), tok_text[0][start_index+1:]), dim=0)
    elif noise_type == 4:
        tok_text = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)['input_ids']
        input_ids = torch.cat((tok_text[0][:start_index+1], torch.LongTensor([tokenizer.mask_token_id]), tok_text[0][start_index+1+num_ops:]), dim=0)
    elif noise_type == 5:
        end_index = num_ops
        tok_text = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)['input_ids']
        input_ids = torch.cat((tok_text[0][:start_index+1], torch.unsqueeze(tok_text[0][end_index+1], 0), tok_text[0][start_index+2:end_index+1], torch.unsqueeze(tok_text[0][start_index+1], 0), tok_text[0][end_index+2:]), dim=0)
        return tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        tok_text = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)['input_ids']
        input_ids = torch.cat((tok_text[0][:start_index+1], tok_text[0][start_index+1+num_ops:]), dim=0)
        return tokenizer.decode(input_ids, skip_special_tokens=True)

    return tokenizer.decode(input_ids, skip_special_tokens=False)

def mbart_generation(batch_text, model, lang_code, tokenizer, device):
    with torch.no_grad():
        batch = tokenizer(batch_text, return_tensors="pt", max_length=128, truncation=True, padding=True)['input_ids'].to(device)
        translated_tokens = model.generate(batch, decoder_start_token_id=tokenizer.lang_code_to_id[lang_code])
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translation

def xlm_roberta_generate(batch_text, model, xlm_tokenizer, device):
    with torch.no_grad():
        input_ids = xlm_tokenizer(batch_text, add_special_tokens = False, return_tensors='pt', max_length=128, truncation=True, padding=True)['input_ids'].to(device)
        logits = model(input_ids).logits

        for i, ele_input_ids in enumerate(input_ids):
            # print(batch_text[i])
            # print(input_ids[i])
            # print((ele_input_ids == xlm_tokenizer.mask_token_id).nonzero())
            masked_index = (ele_input_ids == xlm_tokenizer.mask_token_id).nonzero().item()
            probs = logits[i, masked_index].softmax(dim=0)
            values, predictions = probs.topk(4)
            pred = random.choices(predictions, k=1)[0]
            input_ids[i][masked_index] = pred
        return xlm_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

def text_score_generate(num_var, lang, ref_lines, noise_planner_num, del_noise_lam, mask_noise_lam, device):
    # load in XLM-Roberta model
    xlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    xlm_model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large").to(device)
    xlm_model.eval()
    # load in MBart model and its tokenzier
    mbart_model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="facebook/mbart-large-cc25").to(device)
    mbart_tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang=lang)
    mbart_model.eval()
    # initialize cand_dict_arr, sen_noise_dict, id_sen_dict: key->seg_noise id, value->sentence list
    cand_dict_arr = {}
    id_sen_score_dict = {}
    id_sen_dict = {} # id_sen_dict is a dict containing "score" and "text" fields, "text" field is a list which contains a history of all generated sentences
    for line_index, ref_line in enumerate(ref_lines):
        for i in range(num_var):
            id = str(line_index)+'_'+str(i)
            tok_xlm_ls = xlm_tokenizer.tokenize(ref_line)
            tok_mbart_ls = mbart_tokenizer.tokenize(ref_line)
            # initialize pretraining scheduling scheme using tokenized word lists
            cand_dict_arr[id] = {}
            cand_dict_arr[id]['xlm'] = min(len(tok_xlm_ls), 126)-1-np.array(range(min(len(tok_xlm_ls), 126)))
            cand_dict_arr[id]['mbart'] = min(len(tok_mbart_ls), 125)-1-np.array(range(min(len(tok_mbart_ls), 125)))
            id_sen_dict[id] = {}
            id_sen_dict[id]['score'] = 0
            id_sen_dict[id]['text'] = [ref_line]
            id_sen_score_dict[id] = [ref_line+" [Score: 0]"]
    # determine each sentence with specified number of noises
    sen_noise_dict, max_step = noise_planner(num_var, len(ref_lines), noise_planner_num)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load in mnli model for severity measures
    mnli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
    mnli_model.eval()
    mnli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    m = nn.Softmax(dim=1)

    batch_size_gen = 16
    batch_size_xlm = 128
    batch_size_mnli = 128
    print("Max Step: ", max_step)
    for step in range(1, max_step+1):
        mbart_add_seg_id_ls, mbart_add_start_ls, mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls, xlm_add_seg_id_ls, xlm_add_start_ls, \
        xlm_replace_seg_id_ls, xlm_replace_start_ls, swap_seg_id_ls, swap_start_ls, swap_end_ls, del_seg_id_ls, del_start_ls, del_len_ls, step_noise_dict = noise_schedule(id_sen_dict, step, sen_noise_dict, cand_dict_arr, del_noise_lam, mask_noise_lam)
        # produce the text for generate functions
        mbart_ls, xlm_ls, swap_ls, delete_ls = [], [], [], []
        # construct mbart add dataset
        for id, start_index in zip(mbart_add_seg_id_ls, mbart_add_start_ls):
            mbart_ls.append(data_construct(id_sen_dict[id]['text'][-1], 1, mbart_tokenizer, start_index, 0))
        # construct mbart replace dataset
        for id, start_index, replace_len in zip(mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls):
            mbart_ls.append(data_construct(id_sen_dict[id]['text'][-1], 2, mbart_tokenizer, start_index, replace_len))
        # construct xlm add daatset
        for id, start_index in zip(xlm_add_seg_id_ls, xlm_add_start_ls):
            xlm_ls.append(data_construct(id_sen_dict[id]['text'][-1], 3, xlm_tokenizer, start_index, 0))
        # construct xlm replace dataset
        for id, start_index in zip(xlm_replace_seg_id_ls, xlm_replace_start_ls):
            xlm_ls.append(data_construct(id_sen_dict[id]['text'][-1], 4, xlm_tokenizer, start_index, 1))
        # construct swap dataset
        for id, start_index, end_index in zip(swap_seg_id_ls, swap_start_ls, swap_end_ls):
            swap_ls.append(data_construct(id_sen_dict[id]['text'][-1], 5, xlm_tokenizer, start_index, end_index))
        # construct del dataset
        for id, start_index, del_len in zip(del_seg_id_ls, del_start_ls, del_len_ls):
            delete_ls.append(data_construct(id_sen_dict[id]['text'][-1], 6, xlm_tokenizer, start_index, del_len))
        print("All <mask>/non <mask> datasets are constructed for generation")
        # sentence seg id with corresponding generated texts
        new_seg_ids, new_step_ls, step_score_ls = [], [], []

        new_seg_ids.extend(mbart_add_seg_id_ls) # add in all seg ids for mbart add
        new_seg_ids.extend(mbart_replace_seg_id_ls) # add in all seg ids for mbart replace
        # add all generated mbart add/replace texts into the new_step_ls
        for mbart_batch in batchify(mbart_ls, batch_size_gen):
            # print(mbart_batch)
            mbart_texts = mbart_generation(mbart_batch, mbart_model, lang, mbart_tokenizer, device)
            new_step_ls.extend(mbart_texts)

        new_seg_ids.extend(xlm_add_seg_id_ls) # add in all seg ids for xlm add
        new_seg_ids.extend(xlm_replace_seg_id_ls) # add in all seg ids for xlm replace
        # add all generated xlm add/replace texts into the new_step_ls
        for xlm_batch in batchify(xlm_ls, batch_size_xlm):
            xlm_texts = xlm_roberta_generate(xlm_batch, xlm_model, xlm_tokenizer, device)
            new_step_ls.extend(xlm_texts)

        new_seg_ids.extend(swap_seg_id_ls) # add in all seg ids for swap
        new_step_ls.extend(swap_ls)
        new_seg_ids.extend(del_seg_id_ls) # add in all seg ids for delete
        new_step_ls.extend(delete_ls)

        # update all cand dict arr for add/replace from xlm, swap and delete noises
        cand_dict_arr = add_update_cand_dict(cand_dict_arr, xlm_add_seg_id_ls, xlm_add_start_ls)
        cand_dict_arr = replace_update_cand_dict(cand_dict_arr, xlm_replace_seg_id_ls, xlm_replace_start_ls)
        cand_dict_arr = swap_update_cand_dict(cand_dict_arr, swap_seg_id_ls, swap_start_ls, swap_end_ls)
        cand_dict_arr = delete_update_cand_dict(cand_dict_arr, del_seg_id_ls, del_start_ls, del_len_ls)

        prev_step_ls = prev_ids_sens_extract(id_sen_dict, new_seg_ids)
        print("Finish one step sentence generation!")

        # use MNLI Roberta large model to determine the severities and scores
        for prev_batch, cur_batch in zip(batchify(prev_step_ls, batch_size_mnli), batchify(new_step_ls, batch_size_mnli)):
            temp_scores_ls = severity_measure(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device)
            step_score_ls.extend(temp_scores_ls)

        print("Finish one step MNLI!")

        # update all the sentences and scores in the prev dict
        for id, new_sen, score in zip(new_seg_ids, new_step_ls, step_score_ls):
            new_sen = " ".join(new_sen.split())
            if new_sen not in id_sen_dict[id]['text']:
                id_sen_dict[id]['text'].append(new_sen)
                id_sen_dict[id]['score'] += score
                id_sen_score_dict[id].append(new_sen+f" [Score: {score}, Info: {step_noise_dict[id]}]")
            else:
                print(id_sen_dict[id]['text'])
                print(new_sen)

        print("Finish one step")

    return id_sen_dict, id_sen_score_dict

@click.command()
@click.option('-num_var')
@click.option('-lang')
@click.option('-src')
@click.option('-ref')
@click.option('-save')
def main(num_var, lang, src, ref, save):
    """num_var: specifies number of different variants we create for each segment, lang: language code for model,
    src: source folder, ref: reference folder, save: file to save all the generated noises"""
    # load into reference file
    random.seed(12)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    noise_planner_num, del_noise_lam, mask_noise_lam = 1.5, 1.5, 1.5
    save_name = save+f'_num_{noise_planner_num}_del_{del_noise_lam}_mask_{mask_noise_lam}_xlm_mbart.csv'
    csvfile = open(save_name, 'w')
    csvwriter = csv.writer(csvfile)
    fields = ['src', 'mt', 'ref', 'score']
    csvwriter.writerow(fields)

    segFile = open(f"{save}_zhen_num_{noise_planner_num}_del_{del_noise_lam}_mask_{mask_noise_lam}_xlm_mbart.tsv", 'wt')
    tsv_writer = csv.writer(segFile, delimiter='\t')

    for src_file, ref_file in zip(sorted(list(glob.glob(src+'/*'))), sorted(list(glob.glob(ref+'/*')))):
        ref_lines = open(ref_file, 'r').readlines()
        src_lines = open(src_file, 'r').readlines()
        ref_lines = [" ".join(line[:-1].split()) for line in ref_lines]
        src_lines = [" ".join(line[:-1].split()) for line in src_lines]

        print("Text Preprocessed to remove newline and Seed: 12")

        start = time.time()
        id_sen_dict, id_sen_score_dict = text_score_generate(int(num_var), lang, ref_lines, noise_planner_num, del_noise_lam, mask_noise_lam, device)
        print("Total generated sentences for one subfile: ", len(id_sen_dict))

        for key, value in id_sen_dict.items():
            seg_id = int(key.split('_')[0])
            noise_sen, score = value['text'][-1], value['score'] # the last processed noise sentence
            csvwriter.writerow([src_lines[seg_id], noise_sen, ref_lines[seg_id], score])


        for _, values in id_sen_score_dict.items():
            tsv_writer.writerow(values)

        print(f"Finished in {time.time()-start} seconds")
        print(f"{csvfile} Subfile outputs are saved in regression csv format!")

if __name__ == "__main__":
    main()
