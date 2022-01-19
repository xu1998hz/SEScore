import csv
import re
import matplotlib.pyplot as plt
import statistics
import collections

with open("mqm_newstest2020_ende.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t")
    header = next(rd)
    reg_str = "<v>(.*?)</v>"
    dist = {}
    total=0
    for row in rd:
        category = row[-2]
        if category == "Accuracy/Untranslated text":
            if row[-1] != 'Neutral' and row[-1] != 'no-error':
                res = re.findall(reg_str, row[-3])
                if len(res) > 0:
                    res = res[0]
                    total+=1
                    # only shows the length of the text not the percentage
                    if len(res.split()) in dist:
                        dist[len(res.split())] += 1
                    else:
                        dist[len(res.split())] = 1

    print("total particular erros: ", total)
    od = collections.OrderedDict(sorted(dist.items()))
    print(od)
    print([od[od_ele]/total for od_ele in od])
