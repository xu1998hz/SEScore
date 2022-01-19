import csv

with open("mqm_newstest2020_zhen.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t")
    header = next(rd)
    dist = {}
    for row in rd:
        key = row[0]+'_'+row[1]+'_'+row[2]+'_'+row[3]+'_'+row[4]
        if key not in dist:
            if row[-2] == 'no-error':
                dist[key] = 0
            else:
                dist[key] = 1
        else:
            if row[-2] != 'no-error':
                dist[key] += 1

    value_dict = {}
    for val in dist.values():
        if val in value_dict:
            value_dict[val] += 1
        else:
            value_dict[val] = 1

    print(value_dict)
    val_dict_percent = {}
    sum_val = sum(value_dict.values())

    for key, vals in value_dict.items():
        val_dict_percent[key] = vals/sum_val

    print(val_dict_percent)
