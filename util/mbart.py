from transformers import MBartForConditionalGeneration, MBartTokenizer

model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="facebook/mbart-large-cc25")
model.eval()
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
pre_article = "He will not accept it becasue he didn't like it ."
article = "</s> He will <mask> it becasue he didn't like it . </s>"
batch = tokenizer(article, return_tensors="pt")
print(tokenizer.tokenize(pre_article))
print(pre_article.split())
print(len(pre_article.split()))
translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"])
translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(translation.split())
print(len(translation.split()))
