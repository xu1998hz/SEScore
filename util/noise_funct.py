import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import random
from word2word import Word2word
import string
import copy

class noise_types():
    # all noise functions will return noise sentence based on the noise types and also return the noise region
    def __init__(self, text, lang1, lang2):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-cased')
        self.word_ls = text.split()
        self.dict = Word2word(lang1, lang2)
        self.noise_region = [0] * len(self.word_ls)

    def predict_helper(self, input_txt, top_k):
        inputs = self.tokenizer(input_txt, return_tensors='pt')
        tokenize_ls = self.tokenizer.tokenize(input_txt)
        re
        outputs = self.model(**inputs)
        predictions = outputs[0]
        # currently, predict all the sequences of MASKs at the same time, need to update to beam search later
        sorted_preds, sorted_idx = predictions[0].sort(dim=-1, descending=True)
        sorted_idx = sorted_idx[1:-1]
        # random number choose from topk
        k = random.randint(0, top_k-1)
        predicted_index = [sorted_idx[i, k].item() for i in range(0, len(tokenize_ls))]
        predicted_token = [self.tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(indices[0], indices[-1]+1)]
        print(indices)
        print(f"top k: {k}")
        print(predicted_token)
        return predicted_token

    def punct_helper(self):
        indices = []
        for i, word in enumerate(self.word_ls):
            if word in string.punctuation:
                indices.append(i)
        return indices

    def addition(self, top_k, n_gram):
        # randomly decided insert postions and randomly decide inserted length ranging from 1 to n-gram
        insert_pos = random.randint(0, len(self.word_ls))
        rand_ngram = random.randint(1, n_gram)
        if insert_pos < len(self.word_ls):
            temp = self.word_ls[:insert_pos].copy()
            temp.extend(['[MASK]']*rand_ngram)
            temp.extend(self.word_ls[insert_pos:].copy())
            # construct temp region for groud truth labels
            temp_region = self.noise_region[:insert_pos].copy()
            temp_region.extend([1]*rand_ngram)
            temp_region.extend(self.noise_region[insert_pos:].copy())
        else:
            temp = self.word_ls.copy()
            temp.extend(['[MASK]']*rand_ngram)
            temp_region = self.noise_region.copy()
            temp_region.extend([1]*rand_ngram)

        self.noise_region = temp_region
        input_txt = ' '.join(temp)
        temp[insert_pos:insert_pos+rand_ngram] = self.predict_helper(input_txt, top_k)
        self.word_ls = temp
        print(self.word_ls)
        print(self.noise_region)

    def omission(self):
        # delete one word at a time; return the omission noise text and noise region
        del_pos = random.randint(0, len(self.word_ls)-1)
        print(del_pos)
        new_word_ls = []
        temp_region = []
        if del_pos == len(self.word_ls)-1:
            new_word_ls.extend(self.word_ls[:del_pos].copy())
            temp_region.extend(self.noise_region[:del_pos].copy())
            temp_region[-1] = 1
        elif del_pos == 0:
            new_word_ls.extend(self.word_ls[del_pos+1:].copy())
            temp_region.extend(self.noise_region[del_pos+1:].copy())
            temp_region[0] = 1
        else:
            new_word_ls.extend(self.word_ls[:del_pos].copy())
            new_word_ls.extend(self.word_ls[del_pos+1:].copy())

            temp_region.extend(self.noise_region[:del_pos].copy())
            temp_region.extend(self.noise_region[del_pos+1:].copy())
            temp_region[del_pos-1] = 1
            temp_region[del_pos] = 1

        self.word_ls = new_word_ls
        self.noise_region = temp_region
        print(self.word_ls)
        print(self.noise_region)

    def mistranslate(self, n_gram, top_k):
        n_add = random.randint(1, n_gram)
        mask_pos = random.randint(0, len(self.word_ls)-n_add)
        temp = self.word_ls[:mask_pos].copy()
        temp.extend(['[MASK]'] * n_add)
        temp.extend(self.word_ls[mask_pos+n_add:].copy())
        return self.predict_helper(' '.join(temp), top_k, mask_pos, mask_pos+n_add)

    def untranslate(self, n_gram):
        # translate one word back to the source language; return the untranslated noise text and noise region
        n_add = random.randint(1, n_gram)
        # check the lexical words in the dictionary
        while True:
            try:
                replace_pos = random.randint(0, len(self.word_ls)-n_add)
                tar_text = self.dict(self.word_ls[replace_pos], n_best=1)[0]
                break
            except KeyError:
                print('Lexical words are not in the dictionary')

        temp = self.word_ls[:replace_pos].copy()
        temp.extend(tar_text.split())
        temp.extend(self.word_ls[replace_pos+n_add:].copy())
        return " ".join(temp), [replace_pos, replace_pos+n_add]

    def punctuation(self):
        punct_indices = self.punct_helper()
        noise_indice = random.randint(0, len(punct_indices)-1)
        rand_punct = random.randint(0, len(string.punctuation)-1)
        temp = []
        temp.extend(self.word_ls[:punct_indices[noise_indice]].copy())
        temp.append(string.punctuation[rand_punct])
        temp.extend(self.word_ls[punct_indices[noise_indice]+1:].copy())
        return " ".join(temp), [punct_indices[noise_indice]]

    def switch(self):
        noise_indice = random.randint(1, len(self.word_ls)-2)
        swap_dir = random.randint(0, 1)
        temp = []
        if swap_dir == 0:
            temp.extend(self.word_ls[:noise_indice-1])
            temp.append(self.word_ls[noise_indice])
            temp.append(self.word_ls[noise_indice-1])
            temp.extend(self.word_ls[noise_indice+1:])
            return " ".join(temp), [noise_indice-1, noise_indice]
        else:
            temp.extend(self.word_ls[:noise_indice])
            temp.append(self.word_ls[noise_indice+1])
            temp.append(self.word_ls[noise_indice])
            temp.extend(self.word_ls[noise_indice+2:])
            return " ".join(temp), [noise_indice, noise_indice+1]

    def capitalization(self):
        # replace capitalization for now
        noise_indice = random.randint(0, len(self.word_ls)-1)
        temp = []
        temp.extend(self.word_ls[:noise_indice])
        # first detect if it is capitalized
        if self.word_ls[noise_indice][0].isupper():
            temp.append(self.word_ls[noise_indice][0].lower()+self.word_ls[noise_indice][1:])
        else:
            temp.append(self.word_ls[noise_indice][0].upper()+self.word_ls[noise_indice][1:])
        temp.extend(self.word_ls[noise_indice+1:])
        return " ".join(temp), [noise_indice]

def main():
    text = "he likes climbing the mountain ."
    noises = noise_types(text, 'en', 'de')
    # first determ9ine how many noises to use

    # num_noises = random.randint(1, 7)
    # print(num_noises)
    # noise_indices = random.sample(range(0, 7), num_noises)
    # print(noise_indices)

    # for i in noise_indices:
    #   if i == 0:
    #noises.addition(top_k=1, n_gram=6)
        # elif i == 1:
    noise_sen, noise_region = noises.mistranslate(n_gram=4, top_k=10)
    print(noise_sen)
        # elif i == 2:
        #     noise_sen, noise_region = noises.untranslate(n_gram=1)
        # elif i == 3:
    #noises.omission()
        # elif i == 4:
        #     noise_sen, noise_region = noises.punctuation()
        # elif i == 5:
        #     noise_sen, noise_region = noises.switch()
        # else:
        #     noise_sen, noise_region = noises.capitalization()

    #     text = noise_sen
    #     print(text)
    #     print(noise_region)
    #     print('------------------------')
    #
    # print(text)


if __name__ == '__main__':
    main()
