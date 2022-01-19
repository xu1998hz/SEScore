import nltk
from nltk.tokenize import word_tokenize
from checklist.perturb import Perturb
from checklist.editor import Editor
import time

sent = "PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening. At the start of the crisis, many people likened it to 1982 or 1973, which was reassuring, because both dates refer to classical cyclical downturns."
editor = Editor(language='english')
pos = nltk.pos_tag(nltk.word_tokenize(sent))
start = time.time()

for i in range(len(pos)):
    w, p = pos[i]
    try:
        syn = Editor().synonyms(sent, w)
    except:
        syn = []

    if len(syn) > 0:
        print(w)
        print(syn)
        print('------------------------')

print(time.time()-start)
