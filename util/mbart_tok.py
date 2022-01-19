from transformers import AutoModelForSequenceClassification, AutoTokenizer, MBartForConditionalGeneration, MBartTokenizer
import torch
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="facebook/mbart-large-cc25").to(device)
model.eval()
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang='en_XX')

with torch.no_grad():
    batch = tokenizer(["</s> <mask> visits earth </s>", "</s> <mask> is handling the construction </s>"], return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
    translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id['en_XX'],
                                        do_sample=True,
                                        max_length=128,
                                        top_k=50,
                                        top_p=0.95,
                                        num_return_sequences=10)
    print(translated_tokens)
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

print(translation)
