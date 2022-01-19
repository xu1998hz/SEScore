import torch
from transformers import BertTokenizer, BertModel,BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
input_txt = "[CLS] She did [MASK] understand ."
inputs = tokenizer(input_txt, return_tensors='pt')

model = BertForMaskedLM.from_pretrained('bert-base-cased')

outputs = model(**inputs)
predictions = outputs[0]

sorted_preds, sorted_idx = predictions[0].sort(dim=-1, descending=True)

predicted_index = [sorted_idx[i, 0].item() for i in range(0,7)]
predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(1,7)]
print(predicted_token)
