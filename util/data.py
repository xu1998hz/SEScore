import random
import numpy as np
from transformers import MBartForConditionalGeneration, MBartTokenizer
from fairseq.models.roberta import RobertaModel
import torch
import torch.nn as nn
import click
import csv

# Mbart model needs to return the generated text with modified length
def mbart_generation(text, start_index, label, model, tokenizer, device, lang_code="en_XX", noise_len=0):
    text_ls = text.split()
    new_text_ls = ['</s>']
    new_text_ls.extend(text_ls[:start_index+1])
    new_text_ls.append('<mask>')

    if label == "replace":
        new_text_ls.extend(text_ls[start_index+1+noise_len:])
    else:
        new_text_ls.extend(text_ls[start_index+1:])

    new_text_ls.append('</s>')
    batch = tokenizer(" ".join(new_text_ls), return_tensors="pt").to(device)
    translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id[lang_code])
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    if label == 'replace':
        return translation, len(translation.split()) - (len(text_ls) - noise_len)
    else:
        return translation, len(translation.split()) - len(text_ls)

def delete(text, start_index, noise_len):
    text_ls = text.split()
    new_text_ls = []
    new_text_ls.extend(text_ls[:start_index+1])
    new_text_ls.extend(text_ls[start_index+1+noise_len:])
    return " ".join(new_text_ls)

def mnli_decider(ref, cand, model, m):
    
    tokens_1 = model.encode(ref, cand)
    result_1 = model.predict('mnli', tokens_1)  # 0: contradiction, 1: neutral, 2: entailment
    softmax_result_1 = m(result_1)[0][-1].item()
    alpha1 = softmax_result_1/(1-softmax_result_1)

    tokens_2 = model.encode(cand, ref)
    result_2 = model.predict('mnli', tokens_2)  # 0: contradiction, 1: neutral, 2: entailment
    softmax_result_2 = m(result_2)[0][-1].item()
    alpha2 = softmax_result_2/(1-softmax_result_2)

    #print("Severity value: ", alpha1*alpha2)
    if alpha1*alpha2 > 81:
        return True
    else:
        return False

def noise_data_generate(src_text, inp_text, num_var, csvwriter, lang):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mnli_model = RobertaModel.from_pretrained('roberta.large.mnli', checkpoint_file='model.pt').to(device)
    mnli_model.eval()
    model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="facebook/mbart-large-cc25").to(device)
    model.eval()
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
    m = nn.Softmax()
    iterations=0
    for i in range(num_var):
        error = 0
        text = inp_text
        #print("Original Text: ", text)
        cand_arr = len(text.split())-1-np.array(range(len(text.split())))
        # Random selection of number of noises
        num_noises = random.choices([1, 2, 3, 4, 5], weights=(0.60, 0.20, 0.10, 0.05, 0.05), k=1)[0]
        # schedule three noises for the experiments now
        for i in range(num_noises):
            noise_index = random.choices([1, 2, 3], weights=(1, 1, 1), k=1)[0]
            # this is the addition noise
            if noise_index == 1:
                new_cand_ls = []
                start_index = random.choices(range(cand_arr.shape[0]), k=1)[0]
                # check if noise position and span length fits current noise context
                if cand_arr[start_index] > 0:
                    cand_text, num_adds = mbart_generation(text, start_index=start_index, label='addition', model=model, tokenizer=tokenizer, device=device, lang_code=lang)
                    if cand_text != text:
                        if mnli_decider(text, cand_text, mnli_model, m):
                            error -= 1
                        else:
                            error -= 5
                        # update all the values on the left
                        for index, left_ele in enumerate(cand_arr[:start_index+1]):
                            new_cand_ls.append(min(start_index-index, cand_arr[index]))
                        new_cand_ls.extend([0]*num_adds)
                        new_cand_ls.extend(list(cand_arr[start_index+1:]))
                        cand_arr = np.array(new_cand_ls)
                        text = cand_text
            # this is the delete noise
            elif noise_index == 2:
                new_cand_ls = []
                num_deletes = random.choices([1, 2, 3, 4], weights=(0.50, 0.30, 0.10, 0.10), k=1)[0]
                start_index = random.choices(range(cand_arr.shape[0]), k=1)[0]
                # check if noise position and span length fits current noise context
                if cand_arr[start_index] >= num_deletes and cand_arr[start_index] != 0:
                    # update all the values on the left
                    for index in range(len(cand_arr[:start_index+1])):
                        new_cand_ls.append(min(start_index-index, cand_arr[index]))

                    new_cand_ls.extend(list(cand_arr[start_index+1+num_deletes:]))
                    cand_arr = np.array(new_cand_ls)
                    cand_text = delete(text, start_index, num_deletes)
                    if mnli_decider(text, cand_text, mnli_model, m):
                        error -= 1
                    else:
                        error -= 5
                    text = cand_text
            # this is the replace noise
            else:
                new_cand_ls = []
                num_replace = random.choices([1, 2, 3, 4, 5, 6], weights=(0.50, 0.25, 0.10, 0.05, 0.05, 0.05), k=1)[0]
                start_index = random.choices(range(cand_arr.shape[0]), k=1)[0]
                # check if noise position and span length fits current noise context
                if cand_arr[start_index] >= num_replace and cand_arr[start_index] != 0:
                    cand_text, new_replace = mbart_generation(text, start_index=start_index, label='replace', model=model, tokenizer=tokenizer, device=device, noise_len=num_replace, lang_code=lang)
                    if cand_text != text:
                        if mnli_decider(text, cand_text, mnli_model, m):
                            error -= 1
                        else:
                            error -= 5
                        # update all the values on the left
                        for index in range(len(cand_arr[:start_index+1])):
                            new_cand_ls.append(min(start_index-index, cand_arr[index]))

                        new_cand_ls.extend([0]*new_replace)
                        new_cand_ls.extend(list(cand_arr[start_index+1+num_replace:]))
                        cand_arr = np.array(new_cand_ls)
                        text = cand_text
        if error < 0:
            csvwriter.writerow([src_text, text, inp_text, error])
            iterations+=1

    return iterations


def main():
    # save all the data to a csv file
    random.seed(0)
    csvfile = open('zh_en_data_81.csv', 'w')
    csvwriter = csv.writer(csvfile)
    fields = ['src', 'mt', 'ref', 'score']
    csvwriter.writerow(fields)
    iterations = 0
    #src_file, ref_file = open('newstest2020-zhen-src.zh.txt', 'r'), open('newstest2020-zhen-ref.en.txt', 'r')
    #for src_text, ref_text in zip(src_file, ref_file, ):
    src_text = ""
    ref_text = "He will not accept it because he doesn't like it"
    iterations+=noise_data_generate(src_text, ref_text, 10, csvwriter, 'en_XX')
    print(f"Finished Iterations {iterations}")
    print("Finish the file generation!")

if __name__ == "__main__":
    main()
