import torch
import torch.nn as nn

# Load the model in fairseq
from fairseq.models.roberta import RobertaModel

m = nn.Softmax()
roberta = RobertaModel.from_pretrained('roberta.large.mnli', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

tokens = roberta.encode('he is happily to receive it .', 'he is happily to have and receive it .')
result = roberta.predict('mnli', tokens)  # 0: contradiction, 1: neutral, 2: entailment

print(result)
if result.argmax() == 1:
    print("Result is neutral")

tokens = roberta.encode("will not accept it he didn't like it", "He will not accept it becasue he just didn't like it")
result = roberta.predict('mnli', tokens)  # 0: contradiction, 1: neutral, 2: entailment

softmax_result = m(result)[0][-1].item()

print(result)
print(softmax_result)
print(softmax_result/(1-softmax_result))
print(result.argmax())
