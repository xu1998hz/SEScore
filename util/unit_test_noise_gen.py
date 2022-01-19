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
import time
import string

def noise_sanity_check(cand_arr, del_noise_lam, mask_noise_lam, words_ls):
    # decide noise type upon function called
    noise_type = random.choices([1, 2, 3, 4, 5, 6], weights=(1, 1, 1, 1, 1, 1), k=1)[0]
    start_index = random.choices(range(cand_arr.shape[0]), k=1)[0]
    if noise_type == 1: # this is the addition noise
        # check if noise position and span length fits current noise context
        if cand_arr[start_index] > 0:
            return noise_type, start_index, 1
    elif noise_type == 2: # this is the delete noise
        num_deletes = random.choices([1, 2, 3, 4], weights=poisson.pmf(np.arange(1, 5, 1), mu=del_noise_lam, loc=1), k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr[start_index] >= num_deletes and cand_arr[start_index] != 0:
            return noise_type, start_index, num_deletes
    elif noise_type == 3: # this is the switch noise
        indices = sorted(random.sample(range(cand_arr.shape[0]), 2))
        start_index, end_index = indices[0], indices[1]
        if cand_arr[start_index] > 0 and cand_arr[end_index] > 0:
            return noise_type, start_index, end_index
    elif noise_type == 4: # this is the punctuation noise
        start_index_punct, funct_type = select_punct_start(words_ls)
        start_index = start_index_punct - 1
        if cand_arr[start_index] > 0:
            return noise_type, start_index, funct_type
    elif noise_type == 5: # manipulation of characters in the string
        if cand_arr[start_index] > 0:
            return noise_type, start_index, 1
    else: # this is replace noise which replace words with one [mask] per segment
        num_replace = random.choices([1, 2, 3, 4, 5, 6], weights=poisson.pmf(np.arange(1, 7, 1), mu=mask_noise_lam, loc=1), k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr[start_index] >= num_replace and cand_arr[start_index] != 0:
            return noise_type, start_index, num_replace
    return -1, -1, -1

"""return planned noise combinations for each sentence with num_var variances"""
def noise_planner(num_var, num_texts, lam):
    sen_noise_dict = {}
    max_step = 0
    for sen_index in range(num_texts):
        for noise_index in range(num_var):
            # Random selection of number of noises
            num_noises = random.choices([1, 2, 3, 4, 5], weights=poisson.pmf(np.arange(1, 6, 1), mu=1, loc=1), k=1)[0]
            sen_noise_dict[str(sen_index)+'_'+str(noise_index)] = num_noises
            if num_noises > max_step:
                max_step = num_noises
    return sen_noise_dict, max_step # return in dict: key->segID_noiseID, value->num of noises

"""return the indices of punct with the respective noise type"""
def select_punct_start(words_ls):
    funct_type = funct_type = random.choices(['replace', 'delete', 'insert'], k=1)[0]
    punct_ls = ['!', ',', '-', '.', ":", ";"]
    ind_ls = []
    # build ind and type lists
    for index, word in enumerate(words_ls):
        if word in punct_ls:
            ind_ls.append(index)
    if len(ind_ls) == 0: # you can only insert punct when there is no punct
        ind = random.choices(range(len(words_ls)), k=1)[0]
        return ind, 'insert'
    else:
        if funct_type == 'insert':
            ind = random.choices(range(len(words_ls)), k=1)[0]
        else:
            # print(words_ls)
            # print(ind_ls)
            ind = random.choices(ind_ls, k=1)[0]
        return ind, funct_type

"""Return the sentence which already applied punctuation noises, ind is the location to apply noises on"""
def punct_noise(words_ls, funct_type, ind):
    punct_ls = ['!', ',', '-', '.', ":", ";"]
    punct_dict = {'!': 0, ',': 1 , '-': 2, '.': 3, ':': 4, ";": 5}
    if funct_type == 'replace':
        new_word = random.choices(punct_ls[:punct_dict[words_ls[ind]]]+punct_ls[punct_dict[words_ls[ind]]+1:], k=1)[0]
        new_words_ls = words_ls[:ind] + [new_word] + words_ls[ind+1:]
    elif funct_type == 'delete': # delete exactly ind
        new_words_ls = words_ls[:ind] + words_ls[ind+1:]
    else: # this is Punctuation error for insert
        rand_punct = random.choices(punct_ls, k=1)[0]
        new_words_ls = words_ls[:ind] + [rand_punct] + words_ls[ind:]
    return new_words_ls

"""Return the sentence which already applied spelling noises, ind is the location to apply noises on"""
def word_noise_adder(words_ls, funct_type, lang, ind):
    # alphabets are language specific
    if lang == 'en_XX':
        lower_alphabets = list(string.ascii_lowercase)
        upper_alphabets = list(string.ascii_uppercase)
        numbers = [str(i) for i in range(10)]
        puncts = ['!', ',', '-', '.', ":", ";"]
        lower_letter_dict = dict(zip(lower_alphabets, range(26)))
        upper_letter_dict = dict(zip(upper_alphabets, range(26)))
        num_dict = dict(zip(numbers, range(10)))
        punct_dict = dict(zip(puncts, range(6)))

    if len(words_ls[ind]) == 1: # only delete
        new_word_ls = words_ls[:ind] + words_ls[ind+1:]
    else: # for words have more than one letter, we can apply delete, swap and insert
        if funct_type == 'delete':
            if len(words_ls[ind]) <= 4: # only apply one letter on each word
                letter_ind = random.choices(range(len(words_ls[ind])), k=1)[0]
                if letter_ind == len(words_ls[ind])-1: # handle the edge case of deleting the last letter of the word
                    new_word = words_ls[ind][:letter_ind]
                else:
                    new_word = words_ls[ind][:letter_ind] + words_ls[ind][letter_ind+1:]
            else:
                span = random.choices([1, 2, 3, 4], weights=poisson.pmf(np.arange(1, 5, 1), mu=0.5, loc=1), k=1)[0]
                # print(words_ls)
                # print(ind)
                start_letter = random.choices(range(len(words_ls[ind])-span+1), k=1)[0]
                if start_letter == len(words_ls[ind])-span: # handle the edge case of deleting last span of the word
                    new_word = words_ls[ind][:start_letter]
                else:
                    new_word = words_ls[ind][:start_letter] + words_ls[ind][start_letter+span:]
        elif funct_type == 'replace':
            letter_ind = random.choices(range(len(words_ls[ind])), k=1)[0]
            letter = words_ls[ind][letter_ind]
            if letter.islower():
                new_letter = random.choices(lower_alphabets[:lower_letter_dict[letter]]+lower_alphabets[lower_letter_dict[letter]+1:], k=1)[0]
            elif letter.isupper():
                new_letter = random.choices(upper_alphabets[:upper_letter_dict[letter]]+upper_alphabets[upper_letter_dict[letter]+1:], k=1)[0]
            elif letter.isnumeric():
                new_letter = random.choices(numbers[:num_dict[letter]]+numbers[num_dict[letter]+1:], k=1)[0]
            else:
                new_letter = random.choices(puncts[:punct_dict[letter]]+puncts[punct_dict[letter]+1:], k=1)[0]
            # print(letter)
            # print(letter_dict)
            if letter_ind == len(words_ls[ind])-1: # handle edge case of replace on the last letter of the word
                new_word = words_ls[ind][:letter_ind] + new_letter
            else:
                new_word = words_ls[ind][:letter_ind] + new_letter + words_ls[ind][letter_ind+1:]
        elif funct_type == 'switch': # swap two letters
            letter_inds = sorted(random.sample(range(len(words_ls[ind])), k=2))
            start_ind, end_ind = letter_inds[0], letter_inds[1]
            if end_ind == len(words_ls[ind])-1: # handle edge case of switch on the last letter of the word
                new_word = words_ls[ind][:start_ind] + words_ls[ind][end_ind] + words_ls[ind][start_ind+1:end_ind] + words_ls[ind][start_ind]
            else:
                new_word = words_ls[ind][:start_ind] + words_ls[ind][end_ind] + words_ls[ind][start_ind+1:end_ind] + words_ls[ind][start_ind] + words_ls[ind][end_ind+1:]
        elif funct_type == 'cap': # 60% of times alter the capiltalization of first tokens
            if words_ls[ind][0].isupper():
                new_word = words_ls[ind][0].lower() + words_ls[ind][1:]
            else: # only change lower to upper when sentences do not have upper
                new_word = words_ls[ind][0].upper() + words_ls[ind][1:]
        else: # insert random characters to all possible positions of token
            letter_ind = random.choices(range(len(words_ls[ind])), k=1)[0]
            rand_letter = random.choices(lower_alphabets, k=1)[0]
            new_word = words_ls[ind][:letter_ind] + rand_letter + words_ls[ind][letter_ind:]
        new_word_ls = words_ls[:ind] + [new_word] + words_ls[ind+1:]
    return new_word_ls

"""seq list dict: key is step index, value is a dict of sentences: key is the segID_noiseID. value is the modifed sentence
    dict: key->segID_noiseID, value->[original sentence, noise1, noise2, ... noisek]"""
def noise_schedule(id_sen_dict, step, sen_noise_dict, cand_dict_arr, del_noise_lam, mask_noise_lam, lang):
    add_start_ls, add_seg_id_ls, add_text_ls = [], [], []
    del_seg_id_ls, del_text_ls, del_start_ls, del_len_ls = [], [], [], []
    replace_seg_id_ls, replace_text_ls, replace_start_ls, replace_len_ls = [], [], [], []

    spell_start_ls, spell_text_ls, spell_seg_id_ls = [], [], []
    punct_start_ls, punct_text_ls, punct_noise_type, punct_seg_id_ls = [], [], [], []
    switch_start_ls, switch_text_ls, switch_end_ls, switch_seg_id_ls = [], [], [], []
    step_noise_dict = {}
    for id, num_noises in sen_noise_dict.items():
        # check if the segment has the valid number of noise for current step
        if step <= num_noises:
            words_ls = preprocess(id_sen_dict[id]['text'][-1])
            noise_type, start_index, num_ops = noise_sanity_check(cand_dict_arr[id], del_noise_lam, mask_noise_lam, words_ls)
            if id == '6_2':
                print(words_ls)
                print(len(words_ls))
                print(cand_dict_arr[id])
                print(cand_dict_arr[id].shape)
                print('6_2: ', noise_type)
            # only if random selected error type and error number is valid
            if noise_type != -1:
                # type1: Accuracy/Addition, Fluency/Grammar
                if noise_type == 1:
                    #print(f"id: {id} -> Addition for current step")
                    new_word_ls = ['</s>']
                    new_word_ls.extend(words_ls[:start_index+1])
                    new_word_ls.append('<mask>')
                    new_word_ls.extend(words_ls[start_index+1:])
                    new_word_ls.append('</s>')
                    add_text_ls.append(" ".join(new_word_ls))
                    add_seg_id_ls.append(id)
                    # this data is used for updating cand_dict_arr for the next step
                    add_start_ls.append(start_index)
                    step_noise_dict[id] = ['Addition', start_index]
                # type2: Accuracy/Omission, Fluency/Grammar
                elif noise_type == 2:
                    #print(f"id: {id} -> Omission for current step")
                    new_word_ls = []
                    new_word_ls.extend(words_ls[:start_index+1])
                    new_word_ls.extend(words_ls[start_index+1+num_ops:])
                    del_text_ls.append(" ".join(new_word_ls))
                    del_seg_id_ls.append(id)
                    # this data is used for updating cand_dict_arr for the next step
                    del_start_ls.append(start_index)
                    del_len_ls.append(num_ops)
                    step_noise_dict[id] = ['Delete', start_index, num_ops]
                # type3: Switch Word Order (One swap at a time), Fluency/Grammar, Style/Awkward
                elif noise_type == 3: # this noise will not modify the length of the sentence
                    #print(f"id: {id} -> Switch for current step")
                    end_index = num_ops
                    # print("start: ", start_index)
                    # print("end: ", end_index)
                    # print(len(words_ls))
                    # print(cand_dict_arr[id].shape)
                    new_word_ls = words_ls[:start_index+1] + [words_ls[end_index+1]] + words_ls[start_index+2:end_index+1] + [words_ls[start_index+1]] + words_ls[end_index+2:]
                    switch_start_ls.append(start_index)
                    switch_end_ls.append(end_index)
                    switch_seg_id_ls.append(id)
                    switch_text_ls.append(" ".join(new_word_ls))
                    step_noise_dict[id] = ['Switch', start_index, num_ops]
                # type4: Delete, replace randomly add punctuations (One Punctuation at a time), Fluency/punctuations
                elif noise_type == 4:
                    #print(f"id: {id} -> Punctuation for current step")
                    funct_type = num_ops
                    new_word_ls = punct_noise(words_ls, funct_type, start_index+1)
                    new_text = ['</s>'] + new_word_ls + ['</s>']
                    punct_text_ls.append(" ".join(new_text))
                    punct_seg_id_ls.append(id)
                    punct_start_ls.append(start_index)
                    punct_noise_type.append(funct_type)
                    step_noise_dict[id] = ['Punctuation', start_index, num_ops]
                # type5: Delete span of characters, Swap characters, Insert characters, Upper/Lowever the characters
                elif noise_type == 5: # this noise will not modify the length of the sentence
                    #print(f"id: {id} -> Spelling for current step")
                    funct_type = random.choices(['replace', 'delete', 'insert', 'cap', 'switch'], k=1)[0]
                    new_word_ls = word_noise_adder(words_ls, funct_type, lang, start_index+1)

                    new_text = ['</s>'] + new_word_ls + ['</s>']
                    spell_text_ls.append(" ".join(new_text))
                    spell_seg_id_ls.append(id)
                    spell_start_ls.append(start_index)
                    step_noise_dict[id] = ['Spelling', start_index, num_ops]
                # type6: Accuracy/Mistranslation, Fluency/Grammar
                else:
                    #print(f"id: {id} -> Mistranslation for current step")
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
                    step_noise_dict[id] = ['Replace', start_index, num_ops]

    # seg_id_ls: a list contains all the seg_noise ids, text_ls: contains all the text in the intermediate processes
    return add_seg_id_ls, add_text_ls, add_start_ls, del_seg_id_ls, del_text_ls, del_start_ls, del_len_ls, replace_seg_id_ls, replace_text_ls, replace_start_ls, replace_len_ls, step_noise_dict,\
        punct_start_ls, punct_text_ls, punct_noise_type, punct_seg_id_ls, switch_start_ls, switch_text_ls, switch_end_ls, switch_seg_id_ls, spell_text_ls, spell_start_ls, spell_seg_id_ls

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
        translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id[lang_code])#, do_sample=True, max_length=128, top_k=50, top_p=0.95, num_return_sequences=10)
        #translated_tokens = select_sen_batch(translated_tokens, step_size)
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    trans_len = np.array([len(preprocess(trans)) for trans in translation])
    translation = [' '.join(preprocess(trans)) for trans in translation]
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
    print('--------------before update----------------------------')
    print(cand_dict_arr['6_2'])
    for replace_seg_id, replace_start, replace_new_len, replace_len in zip(replace_seg_id_ls, replace_start_ls, replace_new_len_ls, replace_len_ls):
        new_cand_ls = []
        # update all the values on the left
        for index, _ in enumerate(cand_dict_arr[replace_seg_id][:replace_start+1]):
            new_cand_ls.append(min(replace_start-index, cand_dict_arr[replace_seg_id][index]))
        if replace_seg_id == '6_2':
            print("replace new distance: ", replace_new_len)
        new_cand_ls.extend([0]*replace_new_len)
        new_cand_ls.extend(list(cand_dict_arr[replace_seg_id][replace_start+1+replace_len:]))
        cand_dict_arr[replace_seg_id] = np.array(new_cand_ls)
    print('--------------After update----------------------------')
    print(cand_dict_arr['6_2'])
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

def punct_update_cand_dict(cand_dict_arr, punct_seg_id_ls, punct_start_ls, punct_noise_type):
    for punct_seg_id, punct_start, noise_type in zip(punct_seg_id_ls, punct_start_ls, punct_noise_type):
        new_cand_ls = []
        for index in range(len(cand_dict_arr[punct_seg_id][:punct_start+1])):
            new_cand_ls.append(min(punct_start-index, cand_dict_arr[punct_seg_id][index]))
        if noise_type == 'delete':
            new_cand_ls.extend(list(cand_dict_arr[punct_seg_id][punct_start+2:]))
        elif noise_type == 'replace':
            new_cand_ls += [0] + list(cand_dict_arr[punct_seg_id][punct_start+2:])
        else: # punct start index is actually behind punct
            new_cand_ls += [0] + list(cand_dict_arr[punct_seg_id][punct_start+1:])
        cand_dict_arr[punct_seg_id] = np.array(new_cand_ls)
    return cand_dict_arr

def switch_update_cand_dict(cand_dict_arr, switch_seg_id_ls, switch_start_ls, switch_end_ls):
    for switch_seg_id, switch_start, switch_end in zip(switch_seg_id_ls, switch_start_ls, switch_end_ls):
        new_cand_ls = []
        for index in range(len(cand_dict_arr[switch_seg_id][:switch_start+1])):
            new_cand_ls.append(min(switch_start-index, cand_dict_arr[switch_seg_id][index]))
        new_cand_ls += [0]
        for index in range(len(cand_dict_arr[switch_seg_id][switch_start+2:switch_end+1])):
            new_cand_ls.append(min(switch_end-switch_start-index, cand_dict_arr[switch_seg_id][switch_start+2+index]))
        new_cand_ls += [0] + list(cand_dict_arr[switch_seg_id][switch_end+2:])
        cand_dict_arr[switch_seg_id] = np.array(new_cand_ls)
    return cand_dict_arr

def spell_update_cand_dict(cand_dict_arr, spell_start_ls, spell_seg_id_ls):
    for spell_seg_id, spell_start in zip(spell_seg_id_ls, spell_start_ls):
        new_cand_ls = []
        for index in range(len(cand_dict_arr[spell_seg_id][:spell_start+1])):
            new_cand_ls.append(min(spell_start-index, cand_dict_arr[spell_seg_id][index]))
        new_cand_ls += [0] + list(cand_dict_arr[spell_seg_id][spell_start+2:])
        cand_dict_arr[spell_seg_id] = np.array(new_cand_ls)
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
    # p_not_severe = 2*softmax_result_1*softmax_result_2/(softmax_result_1+softmax_result_2)
    scores = []
    for prob_1, prob_2 in zip(softmax_result_1, softmax_result_2):
        if prob_1 > 0.9 and prob_2 > 0.9:
            scores.append(-1)
        else:
            scores.append(-5)
    return scores

def text_score_generate(num_var, lang, ref_lines, planner_lam, del_noise_lam, mask_noise_lam):
    # initialize cand_dict_arr, sen_noise_dict, id_sen_dict: key->seg_noise id, value->sentence list
    cand_dict_arr = {}
    id_sen_dict = {}
    id_sen_score_dict = {}
    for line_index, ref_line in enumerate(ref_lines):
        for i in range(num_var):
            id = str(line_index)+'_'+str(i)
            process_word_ls = preprocess(ref_line)
            cand_dict_arr[id] = len(process_word_ls)-1-np.array(range(len(process_word_ls)))
            id_sen_dict[id] = {}
            id_sen_dict[id]['score'] = 0
            id_sen_dict[id]['text'] = [ref_line]
            id_sen_score_dict[id] = [ref_line+" [Score: 0]"]
    # print("Step 0: initialization of noise scheduling scheme")
    # print(cand_dict_arr)
    # print('----------------------------------------------------------------')
    sen_noise_dict, max_step = noise_planner(num_var, len(ref_lines), planner_lam)
    # print("Noise Planner: ")
    # print(sen_noise_dict)

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
        add_seg_id_ls, add_text_ls, add_start_ls, del_seg_id_ls, del_text_ls, del_start_ls, del_len_ls, replace_seg_id_ls, replace_text_ls, replace_start_ls, replace_len_ls, step_noise_dict, \
        punct_start_ls, punct_text_ls, punct_noise_type, punct_seg_id_ls, switch_start_ls, switch_text_ls, switch_end_ls, switch_seg_id_ls, spell_text_ls, spell_start_ls, spell_seg_id_ls = noise_schedule(id_sen_dict, step, sen_noise_dict, cand_dict_arr, del_noise_lam, mask_noise_lam, lang)
        new_step_ls, new_seg_ids = [], []
        add_new_len_ls, replace_new_len_ls = [], []
        # add all ids, this has to be handled with care, all string manipulation goes first
        new_seg_ids.extend(del_seg_id_ls)
        new_seg_ids.extend(punct_seg_id_ls)
        new_seg_ids.extend(switch_seg_id_ls)
        new_seg_ids.extend(spell_seg_id_ls)
        # those two need to be generated by MBart
        new_seg_ids.extend(add_seg_id_ls)
        new_seg_ids.extend(replace_seg_id_ls)
        # add delete generated texts (this has to be handled with care, all string manipulation goes first)
        new_step_ls.extend(del_text_ls)
        new_step_ls.extend(punct_text_ls)
        new_step_ls.extend(switch_text_ls)
        new_step_ls.extend(spell_text_ls)
        step_score_ls = []
        # addition generated texts as well as its respective modified length
        for add_batch in batchify(add_text_ls, batch_size_gen):
            #print("Add batch length: ", len(add_batch))
            add_batch_ls, add_batch_len = mbart_generation(add_batch, model, tokenizer, device, lang, step_size)
            new_step_ls.extend(add_batch_ls)
            add_new_len_ls.extend(add_batch_len)
        # update cand_dict_arr for add operation
        cand_dict_arr = add_update_cand_dict(cand_dict_arr, add_seg_id_ls, add_start_ls, add_new_len_ls)
        # replace generated texts as well as its respective modified length
        for replace_batch, replace_len_batch in zip(batchify(replace_text_ls, batch_size_gen), batchify(replace_len_ls, batch_size_gen)):
            #print("replace batch length: ", len(replace_batch))
            replace_batch_ls, replace_batch_len = mbart_generation(replace_batch, model, tokenizer, device, lang, step_size)
            new_step_ls.extend(replace_batch_ls)
            replace_new_len_ls.extend(replace_batch_len)
        # update cand_dict_arr for replace operation
        cand_dict_arr = replace_update_cand_dict(cand_dict_arr, replace_seg_id_ls, replace_start_ls, replace_new_len_ls, replace_len_ls)
        # update cand_dict_arr for delete operation
        cand_dict_arr = delete_update_cand_dict(cand_dict_arr, del_seg_id_ls, del_start_ls, del_len_ls)
        # update cand dict arr for punct operation
        cand_dict_arr = punct_update_cand_dict(cand_dict_arr, punct_seg_id_ls, punct_start_ls, punct_noise_type)
        # update cand dict arr for switch operation
        cand_dict_arr = switch_update_cand_dict(cand_dict_arr, switch_seg_id_ls, switch_start_ls, switch_end_ls)
        # update cand dict arr fpr spelling operation
        cand_dict_arr = spell_update_cand_dict(cand_dict_arr, spell_start_ls, spell_seg_id_ls)
        # print(f"Current Step {step}: ")
        # print(cand_dict_arr)
        # print('-------------------------------------------------')
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
            new_sen = " ".join(new_sen.split())
            if new_sen not in id_sen_dict[id]['text']:
                id_sen_dict[id]['text'].append(new_sen)
                id_sen_dict[id]['score'] += score
                if id == "6_2":
                    print("Before save to dict: ")
                    print(new_sen)
                id_sen_score_dict[id].append(new_sen+f" [Score: {score}, Info: {step_noise_dict[id]}]")
                # print(f"Current ID: {id}")
                # print(f"Current text: {new_sen}")
                # print('------------------------------------------------------')
                # print(f"Current scores: {score}")
                # print('------------------------------------------------------')

        print("Finish one step")

    return id_sen_dict, id_sen_score_dict

def preprocess(text):
    return re.findall(r"[\w]+|[^\s\w]", text)

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
    ref_lines = open(ref, 'r').readlines()
    src_lines = open(src, 'r').readlines()
    ref_lines = [" ".join(line[:-1].split()) for line in ref_lines]
    src_lines = [" ".join(line[:-1].split()) for line in src_lines]

    random.seed(12)
    print("Text Preprocessed to remove newline and Seed: 12")
    # for noise_planner in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
    #     for del_noise_lam in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
    #         for mask_noise_lam in [0.5, 0.75]: # , 1, 1.25, 1.5, 1.75, 2, 2.5
    noise_planner, del_noise_lam, mask_noise_lam = 1, 1.5, 1.5
    save_name = save+f'_num_{noise_planner}_del_{del_noise_lam}_mask_{mask_noise_lam}.csv'
    csvfile = open(save_name, 'w')
    csvwriter = csv.writer(csvfile)
    fields = ['src', 'mt', 'ref', 'score']
    csvwriter.writerow(fields)

    start = time.time()
    id_sen_dict, id_sen_score_dict = text_score_generate(int(num_var), lang, ref_lines, noise_planner, del_noise_lam, mask_noise_lam)
    print("Total generated sentences for one subfile: ", len(id_sen_dict))

    for key, value in id_sen_dict.items():
        seg_id = int(key.split('_')[0])
        noise_sen, score = value['text'][-1], value['score'] # the last processed noise sentence
        csvwriter.writerow([src_lines[seg_id], noise_sen, ref_lines[seg_id], score])

    segFile = open(f"{save}_zhen_num_{noise_planner}_del_{del_noise_lam}_mask_{mask_noise_lam}.tsv", 'wt')
    tsv_writer = csv.writer(segFile, delimiter='\t')
    for _, values in id_sen_score_dict.items():
        tsv_writer.writerow(values)

    print(f"Finished in {time.time()-start} seconds")
    print(f"{csvfile} Subfile outputs are saved in regression csv format!")

if __name__ == "__main__":
    main()
