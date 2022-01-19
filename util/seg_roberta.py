from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import click

src = "From September 26 to 27, in order to deepen and expand the educational achievements of the theme of “Remain true to our original aspiration and keep our mission firmly in mind”, the Central Military-Civilian Integration Office, the State Administration of Science and Industry for National Defense, and the All-China Federation of Industry and Commerce jointly organized the research event of “state-owned enterprises and advantageous private enterprises entering the revolutionary base areas in southern Jiangxi"" to review the prospect of revolution, follow the red footprints, and pass on the red genes, thus the purpose of this activity is to interface with the needs of related enterprises and Ganzhou City and to accelerate the development of revolutionary areas."
ref = "From September 26 to 27, in order to deepen and expand the educational achievements of the theme of “Remain true to our original aspiration and keep our mission firmly in mind”, the Central Military-Civilian Integration Office, the State Administration of Science and Industry for National Defense, and the All-China Federation of Industry and Commerce jointly organized the research event of “state-owned enterprises and advantageous private enterprises entering the revolutionary base areas in southern Jiangxi"" to review the process of generation, follow the red footprints, and pass on the red genes, thus the purpose of this activity is to interface with the needs of related enterprises and Ganzhou City and to accelerate the development of revolutionary areas."

model_name = "roberta-large-mnli"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Download pytorch model
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
m = torch.nn.Softmax(dim=1)
with torch.no_grad():
    inputs = tokenizer(src, ref, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
    outputs = model(**inputs).logits
    softmax_result_1 = m(outputs)[:, -1]

    print("Ref entails mt: ", softmax_result_1)

    inputs = tokenizer(ref, src, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
    outputs = model(**inputs).logits
    softmax_result_2 = m(outputs)[:, -1]

    print("Mt entails ref: ", softmax_result_2)
 
