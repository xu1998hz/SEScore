from transformers import AutoModelForSequenceClassification, AutoTokenizer, MBartForConditionalGeneration, MBartTokenizer

model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
TXT = "</s> My friends are <mask> <mask> <mask> they eat too many carbs. </s> en_XX"
input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors='pt')['input_ids']
print(input_ids)
logits = model(input_ids).logits
masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero()

# print(masked_index[0].item())
# probs = logits[0, masked_index[0].item()].softmax(dim=0)
# values, predictions = probs.topk(5)
# print(values)
# print(predictions)
# print(tokenizer.decode(predictions).split())
#
# print(masked_index[1].item())

# probs = logits[0, masked_index].softmax(dim=0)
# values, predictions = probs.topk(5)
# print(values)
# print(predictions)
# print(tokenizer.decode(predictions).split())
