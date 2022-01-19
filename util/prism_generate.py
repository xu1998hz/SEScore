import torch
from transformers import BartForConditionalGeneration, BartTokenizer

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')

model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

inp_lines = open("train/train.ref.zhen", 'r').readlines()
saveFile = open("jan_9_zhen_paraphrase.txt", 'w')

for batch in batchify(inp_lines, 16):
    text = [line[:-1] for line in batch]
    batch = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    generated_ids = model.generate(batch['input_ids'], max_length=128, do_sample=True, top_k=120, top_p=0.95, num_return_sequences=5)

    generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for output in generated_sentence:
        saveFile.write(output+'\n')
        
