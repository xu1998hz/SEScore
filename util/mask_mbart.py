from transformers import MBartTokenizer, MBartForConditionalGeneration
tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
# de_DE is the language symbol id <LID> for German
TXT = "</s> From September 26 to 27, in order to deepen and expand the educational achievements of the theme of “Remain true to our original aspiration and keep our mission firmly in mind”, the Central Military-Civilian Integration Office, the State Administration of Science and Industry for National Defense, and the All-China Federation of Industry and Commerce jointly organized the research event of “state-owned enterprises and advantageous private enterprises entering the revolutionary base areas in southern Jiangxi"" to review the <mask> <mask> <mask>, follow the red footprints, and pass on the red genes, thus the purpose of this activity is to interface with the needs of related enterprises and Ganzhou City and to accelerate the development of revolutionary areas. </s> en_XX"
print(tokenizer.tokenize(TXT))

model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors='pt')['input_ids']
logits = model(input_ids).logits

masked_indexes = (input_ids[0] == tokenizer.mask_token_id).nonzero()
for masked_index in masked_indexes:
    masked_index = masked_index.item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(16)
    print(values)
    print(predictions)
    print(tokenizer.decode(predictions))
    print(tokenizer.decode(predictions).split())
