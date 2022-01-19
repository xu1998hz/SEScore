import random
import click
from transformers import AutoModelForSequenceClassification, AutoTokenizer, MBartForConditionalGeneration, MBartTokenizer
import torch
import csv
import numpy as np
import torch.nn as nn
import glob
import re
from scipy.stats import poisson

def noise_sanity_check(cand_arr):
    # decide noise type upon function called
    noise_type = random.choices([1, 2, 3], weights=(1, 1, 1), k=1)[0]
    start_index = random.choices(range(cand_arr.shape[0]), k=1)[0]
    # this is the addition noise
    if noise_type == 1:
        # check if noise position and span length fits current noise context
        if cand_arr[start_index] > 0:
            return noise_type, start_index, 1
    # this is the delete noise
    elif noise_type == 2:
        num_deletes = random.choices([1, 2, 3, 4], weights=[0.50, 0.30, 0.10, 0.10], k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr[start_index] >= num_deletes and cand_arr[start_index] != 0:
            return noise_type, start_index, num_deletes
    else:
        num_replace = random.choices([1, 2, 3, 4, 5, 6], weights=[0.50, 0.25, 0.10, 0.05, 0.05, 0.05], k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr[start_index] >= num_replace and cand_arr[start_index] != 0:
            return noise_type, start_index, num_replace
    return -1, -1, -1

"""return planned noise combinations for each sentence with num_var variances"""
def noise_planner(num_var, num_texts):
    sen_noise_dict = {}
    max_step = 0
    for sen_index in range(num_texts):
        for noise_index in range(num_var):
            # Random selection of number of noises
            num_noises = random.choices([1, 2, 3, 4, 5], weights=[0.60, 0.20, 0.10, 0.05, 0.05], k=1)[0]
            sen_noise_dict[str(sen_index)+'_'+str(noise_index)] = num_noises
            if num_noises > max_step:
                max_step = num_noises
    return sen_noise_dict, max_step # return in dict: key->segID_noiseID, value->num of noises (A list of noise types)

"""seq list dict: key is step index, value is a dict of sentences: key is the segID_noiseID. value is the modifed sentence
    dict: key->segID_noiseID, value->[original sentence, noise1, noise2, ... noisek]"""
def noise_schedule(id_sen_dict, step, sen_noise_dict, cand_dict_arr):
    add_seg_id_ls, add_text_ls, del_seg_id_ls, del_text_ls, replace_seg_id_ls, replace_text_ls = [], [], [], [], [], []
    add_start_ls, del_start_ls, del_len_ls, replace_start_ls, replace_len_ls = [], [], [], [], []
    for id, num_noises in sen_noise_dict.items():
        # check if the segment has the valid number of noise for current step
        if step <= num_noises:
            noise_type, start_index, num_ops = noise_sanity_check(cand_dict_arr[id])
            # only if random selected error type and error number is valid
            if noise_type != -1:
                # most recent noise sentence
                words_ls = id_sen_dict[id]['text'][-1].split()
                # type1: Accuracy/Addition, Fluency/Grammar
                if noise_type == 1:
                    new_word_ls = ['</s>']
                    new_word_ls.extend(words_ls[:start_index+1])
                    new_word_ls.append('<mask>')
                    new_word_ls.extend(words_ls[start_index+1:])
                    new_word_ls.append('</s>')
                    add_text_ls.append(" ".join(new_word_ls))
                    add_seg_id_ls.append(id)
                    # this data is used for updating cand_dict_arr for the next step
                    add_start_ls.append(start_index)
                # type2: Accuracy/Omission, Fluency/Grammar
                elif noise_type == 2:
                    new_word_ls = []
                    new_word_ls.extend(words_ls[:start_index+1])
                    new_word_ls.extend(words_ls[start_index+1+num_ops:])
                    del_text_ls.append(" ".join(new_word_ls))
                    del_seg_id_ls.append(id)
                    # this data is used for updating cand_dict_arr for the next step
                    del_start_ls.append(start_index)
                    del_len_ls.append(num_ops)
                # # type3: Switch Word Order, Fluency/Grammar, Style/Awkward
                # elif noise_type == 3:
                #     pass
                # # type4: Delete, replace randomly add punctuations, Fluency/punctuations
                # elif noise_type == 4:
                #     pass
                # # type5: Delete span of characters, Swap characters, Insert characters, Upper/Lowever the characters
                # elif noise_type == 5:
                #     pass
                else:
                    new_word_ls = ['</s>']
                    new_word_ls.extend(words_ls[:start_index+1])
                    new_word_ls.append('<mask>')
                    new_word_ls.extend(words_ls[start_index+1+num_ops:])
                    new_word_ls.append('</s>')
                    replace_text_ls.append(" ".join(new_word_ls))
                    replace_seg_id_ls.append(id)
                    # this data is used for updating cand_dict_arr for the next step
                    replace_start_ls.append(start_index)
                    replace_len_ls.append(num_ops)

    # seg_id_ls: a list contains all the seg_noise ids, text_ls: contains all the text in the intermediate processes
    return add_seg_id_ls, add_text_ls, add_start_ls, del_seg_id_ls, del_text_ls, del_start_ls, del_len_ls, replace_seg_id_ls, replace_text_ls, replace_start_ls, replace_len_ls

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

def select_sen_batch(translated_tokens, step_size):
    batch_return = []
    for i in range(0, translated_tokens.size(0), step_size):
        index = random.choices(range(i, i+step_size), k=1)[0]
        batch_return.append(translated_tokens[index])
    return batch_return

"""Mbart Generation for both addiiton noises and replace noises"""
def mbart_generation(batch_text, model, tokenizer, device, lang_code, step_size):
    batch_len = np.array([len(text.split())-3 for text in batch_text])
    with torch.no_grad():
        batch = tokenizer(batch_text, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
        translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id[lang_code])
                                            # do_sample=True,
                                            # max_length=128,
                                            # top_k=50,
                                            # top_p=0.95,
                                            # num_return_sequences=10)

        #translated_tokens = select_sen_batch(translated_tokens, step_size)
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    trans_len = np.array([len(trans.split()) for trans in translation])

    return translation, trans_len - batch_len

"""add operation to update the candidate dict"""
def add_update_cand_dict(cand_dict_arr, add_seg_id_ls, add_start_ls, add_new_len_ls):
    for add_seg_id, add_start, add_new_len in zip(add_seg_id_ls, add_start_ls, add_new_len_ls):
        new_cand_ls = []
        # update all the values on the left
        for index, _ in enumerate(cand_dict_arr[add_seg_id][:add_start+1]):
            new_cand_ls.append(min(add_start-index, cand_dict_arr[add_seg_id][index]))
        new_cand_ls.extend([0]*add_new_len)
        new_cand_ls.extend(list(cand_dict_arr[add_seg_id][add_start+1:]))
        cand_dict_arr[add_seg_id] = np.array(new_cand_ls)
    return cand_dict_arr

"""replace operation to update the candidate dict"""
def replace_update_cand_dict(cand_dict_arr, replace_seg_id_ls, replace_start_ls, replace_new_len_ls, replace_len_ls):
    for replace_seg_id, replace_start, replace_new_len, replace_len in zip(replace_seg_id_ls, replace_start_ls, replace_new_len_ls, replace_len_ls):
        new_cand_ls = []
        # update all the values on the left
        for index, _ in enumerate(cand_dict_arr[replace_seg_id][:replace_start+1]):
            new_cand_ls.append(min(replace_start-index, cand_dict_arr[replace_seg_id][index]))
        new_cand_ls.extend([0]*replace_new_len)
        new_cand_ls.extend(list(cand_dict_arr[replace_seg_id][replace_start+1+replace_len:]))
        cand_dict_arr[replace_seg_id] = np.array(new_cand_ls)
    return cand_dict_arr

"""delete operation to update the candidate dict"""
def delete_update_cand_dict(cand_dict_arr, del_seg_id_ls, del_start_ls, del_len_ls):
    for del_seg_id, del_start, del_len in zip(del_seg_id_ls, del_start_ls, del_len_ls):
        new_cand_ls = []
        # update all the values on the left
        for index in range(len(cand_dict_arr[del_seg_id][:del_start+1])):
            new_cand_ls.append(min(del_start-index, cand_dict_arr[del_seg_id][index]))
        new_cand_ls.extend(list(cand_dict_arr[del_seg_id][del_start+1+del_len:]))
        cand_dict_arr[del_seg_id] = np.array(new_cand_ls)
    return cand_dict_arr

def prev_ids_sens_extract(id_sen_dict, new_seg_ids):
    prev_sen_ls = []
    for id in new_seg_ids:
        prev_sen_ls.append(id_sen_dict[id]['text'][-1])
    return prev_sen_ls

def severity_measure(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device):
    with torch.no_grad():
        inputs_1 = mnli_tokenizer(prev_batch, cur_batch, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
        output_1 = mnli_model(**inputs_1).logits # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_1 = m(output_1)[:, -1]

        inputs_2 = mnli_tokenizer(cur_batch, prev_batch, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
        output_2 = mnli_model(**inputs_2).logits # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_2 = m(output_2)[:, -1]

    # Use the harmonic mean for threshold, threshold = 0.9
    p_not_severe = softmax_result_1/(1-softmax_result_1) * softmax_result_2/(1-softmax_result_2)
    scores = []
    for prob in p_not_severe:
        if prob > 81:
            scores.append(-1)
        else:
            scores.append(-5)
    return scores

def text_score_generate(num_var, lang, ref_lines):
    # initialize cand_dict_arr, sen_noise_dict, id_sen_dict: key->seg_noise id, value->sentence list
    cand_dict_arr = {}
    id_sen_dict = {} # id_sen_dict is a dict containing "score" and "text" fields, "text" field is a list which contains a history of all generated sentences
    for line_index, ref_line in enumerate(ref_lines):
        for i in range(num_var):
            id = str(line_index)+'_'+str(i)
            cand_dict_arr[id] = len(ref_line.split())-1-np.array(range(len(ref_line.split())))
            id_sen_dict[id] = {}
            id_sen_dict[id]['score'] = 0
            id_sen_dict[id]['text'] = [ref_line]
            #id_orisen_dict[id] = ref_line
    sen_noise_dict, max_step = noise_planner(num_var, len(ref_lines))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="facebook/mbart-large-cc25").to(device)
    model.eval()
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang=lang)

    mnli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
    mnli_model.eval()
    mnli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    m = nn.Softmax(dim=1)
    batch_size_gen = 16
    batch_size_mnli = 16
    step_size = 10
    print("Max Step: ", max_step)
    for step in range(1, max_step+1):
        # del_text_ls is already processed text
        add_seg_id_ls, add_text_ls, add_start_ls, del_seg_id_ls, del_text_ls, del_start_ls, del_len_ls, replace_seg_id_ls, replace_text_ls, replace_start_ls, replace_len_ls = noise_schedule(id_sen_dict, step, sen_noise_dict, cand_dict_arr)
        new_step_ls, new_seg_ids = [], []
        add_new_len_ls, replace_new_len_ls = [], []
        # add all ids
        new_seg_ids.extend(del_seg_id_ls)
        new_seg_ids.extend(add_seg_id_ls)
        new_seg_ids.extend(replace_seg_id_ls)
        # add delete generated texts
        new_step_ls.extend(del_text_ls)
        step_score_ls = []
        # addition generated texts as well as its respective modified length
        for add_batch in batchify(add_text_ls, batch_size_gen):
            add_batch_ls, add_batch_len = mbart_generation(add_batch, model, tokenizer, device, lang, step_size)
            new_step_ls.extend(add_batch_ls)
            add_new_len_ls.extend(add_batch_len)
        # update cand_dict_arr for add operation
        cand_dict_arr = add_update_cand_dict(cand_dict_arr, add_seg_id_ls, add_start_ls, add_new_len_ls)
        # replace generated texts as well as its respective modified length
        for replace_batch, replace_len_batch in zip(batchify(replace_text_ls, batch_size_gen), batchify(replace_len_ls, batch_size_gen)):
            replace_batch_ls, replace_batch_len = mbart_generation(replace_batch, model, tokenizer, device, lang, step_size)
            new_step_ls.extend(replace_batch_ls)
            replace_new_len_ls.extend(replace_batch_len)
        # update cand_dict_arr for replace operation
        cand_dict_arr = replace_update_cand_dict(cand_dict_arr, replace_seg_id_ls, replace_start_ls, replace_new_len_ls, replace_len_ls)
        # update cand_dict_arr for delete operation
        cand_dict_arr = delete_update_cand_dict(cand_dict_arr, del_seg_id_ls, del_start_ls, del_len_ls)
        # extract out all the sentences that need to be compared from the last step
        prev_step_ls = prev_ids_sens_extract(id_sen_dict, new_seg_ids)

        print("Finish one step sentence generation!")
        # use MNLI Roberta large model to determine the severities and scores
        for prev_batch, cur_batch in zip(batchify(prev_step_ls, batch_size_mnli), batchify(new_step_ls, batch_size_mnli)):
            temp_scores_ls = severity_measure(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device)
            step_score_ls.extend(temp_scores_ls)

        print("Finish one step MNLI!")
        # update all the sentences and scores in the prev dict
        for id, new_sen, score in zip(new_seg_ids, new_step_ls, step_score_ls):
            if new_sen not in id_sen_dict[id]['text']:
                id_sen_dict[id]['text'].append(new_sen)
                id_sen_dict[id]['score'] += score

        print("Finish one step")

    return id_sen_dict

def preprocess(text_ls):
    new_text_ls = []
    for text in text_ls:
        new_text_ls.append(' '.join(re.findall(r"[\w]+|[^\s\w]", text[:-1])))
    return new_text_ls

@click.command()
@click.option('-num_var')
@click.option('-lang')
@click.option('-src')
@click.option('-ref')
@click.option('-save')
def main(num_var, lang, src, ref, save):
    """num_var: specifies number of different variants we create for each segment, lang: language code for model,
    src: source folder, ref: reference folder, save: file to save all the generated noises"""
    # load in the file to save data
    csvfile = open(save, 'w')
    csvwriter = csv.writer(csvfile)
    fields = ['src', 'mt', 'ref', 'score']
    csvwriter.writerow(fields)
    random.seed(12)
    # load into reference file
    for src_file, ref_file in zip(sorted(list(glob.glob(src+'/*'))), sorted(list(glob.glob(ref+'/*')))):
        #assert src_file.split('_')[0] == ref_file.split('_')[0]
        print(ref_file)
        ref_lines = open(ref_file, 'r').readlines()
        src_lines = open(src_file, 'r').readlines()
        # preprocess the newline and separate punctuations
        # process_ref_lines = preprocess(ref_lines)
        ref_lines = [line[:-1] for line in ref_lines]
        src_lines = [line[:-1] for line in src_lines]
        print("Finish text preprocessing!")

        id_sen_dict = text_score_generate(int(num_var), lang, ref_lines)
        print('-----------------------------------------------------------')
        print("Total generated sentences for one subfile: ", len(id_sen_dict))

        for key, value in id_sen_dict.items():
            seg_id = int(key.split('_')[0])
            noise_sen, score = value['text'][-1], value['score'] # the last processed noise sentence
            csvwriter.writerow([src_lines[seg_id], noise_sen, ref_lines[seg_id], score])

        print("Subfile outputs are saved in regression csv format!")

if __name__ == "__main__":
    main()
