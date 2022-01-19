import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, MBartForConditionalGeneration, MBartTokenizer
from transformers import AutoModelForMaskedLM
import time
import numpy as np
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
lang='de_DE'
component = 'mbert'

if component == 'mbart':
    model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="facebook/mbart-large-cc25").to(device)
    model.eval()
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang=lang)
    max_len_seq = '<mask> '*128
    max_batch = [max_len_seq] * 16
    # benchmark the thearatical time for mbart
    with torch.no_grad():
        start = time.time()
        for i in range(1000):
            batch = tokenizer(max_batch, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
            translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id[lang])
        print(f"Finished in {time.time()-start} s")

elif component == 'roberta':
    # mnli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
    # mnli_model.eval()
    mnli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    max_len_seq = '<mask> '*128
    max_batch = [max_len_seq] * 16
    inputs_1 = mnli_tokenizer(max_batch, max_batch, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
    while True:
        print(inputs_1)
    # m = nn.Softmax(dim=1)
    # # benchmark the thearatical time for Roberta-Mnli
    # with torch.no_grad():
    #     start = time.time()
    #     for i in range(1000):
    #         with torch.no_grad():
    #
    #             output_1 = mnli_model(**inputs_1).logits # 0: contradiction, 1: neutral, 2: entailment
    #             softmax_result_1 = m(output_1)[:, -1]
    #             alpha1 = softmax_result_1/(1-softmax_result_1)
    #
    #             inputs_2 = mnli_tokenizer(max_batch, max_batch, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
    #             output_2 = mnli_model(**inputs_2).logits # 0: contradiction, 1: neutral, 2: entailment
    #             softmax_result_2 = m(output_2)[:, -1]
    #             alpha2 = softmax_result_2/(1-softmax_result_2)
    #             print("Finish One iter")
        #print(f"Finished in {time.time()-start} s")
else:
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large").to(device)
    model.eval()
    max_len_seq = '<mask> '*128
    max_batch = [max_len_seq] * 128
    with torch.no_grad():
        start = time.time()
        for i in range(10000):
            encoded_input = tokenizer(max_batch, return_tensors='pt', max_length=128, truncation=True, padding=True).to(device)
            output = model(**encoded_input)
            print(output)
        print(f"Finished in {time.time()-start} s")
