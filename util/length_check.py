import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, MBartForConditionalGeneration, MBartTokenizer

inFile = open('eval-zhen-ref.txt', 'r')
# load in both mbart and roberta mnli to check maximum tokenized length
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang='en_XX')
mnli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

max_bart_length = 0
max_mnli_length = 0
bart_len_ls, mnli_len_ls = [], []
for row in inFile:
    cur_bart_len, cur_mnli_len = len(tokenizer.tokenize(row)), len(mnli_tokenizer.tokenize(row))
    bart_len_ls.append(cur_bart_len)
    mnli_len_ls.append(cur_mnli_len)
    if cur_bart_len > max_bart_length:
        max_bart_length = cur_bart_len

    if cur_mnli_len > max_mnli_length:
        max_mnli_length = cur_mnli_len

print("Max Bart Length: ", max_bart_length)
print("Max MNLI Length", max_mnli_length)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('MBart vs MNLI tokenized length dist')
ax1.hist(bart_len_ls, 100, histtype ='bar')
ax2.hist(mnli_len_ls, 100, histtype ='bar')
plt.show()
