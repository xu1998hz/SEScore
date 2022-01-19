import csv
import re
import matplotlib.pyplot as plt
import statistics
import collections

with open("mqm_newstest2020_ende.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t")
    header = next(rd)
    reg_str = "<v>(.*?)</v>"
    seg_type_dict = {}
    for row in rd:
        loc = row[0]+'_'+row[1]+'_'+row[2]+'_'+row[3]
        if row[-1] != 'Neutral' and row[-1] != 'no-error':
            res = re.findall(reg_str, row[-3])
            if len(res) > 0:
                res = res[0]
                span_len = len(res.split())
            else:
                span_len = 0

            if loc in seg_type_dict:
                seg_type_dict[loc]['num_errors'] += 1
                seg_type_dict[loc]['span_length'] += span_len
            else:
                seg_type_dict[loc] = {}
                seg_type_dict[loc]['num_errors'] = 1
                seg_type_dict[loc]['span_length'] = span_len

    dist_len = {}
    dist_errs = {}
    for key, value in seg_type_dict.items():
        if value['num_errors'] in dist_errs:
            dist_errs[value['num_errors']] += 1
        else:
            dist_errs[value['num_errors']] = 1
        if value['span_length'] in dist_len:
            dist_len[value['span_length']] += 1
        else:
            dist_len[value['span_length']] = 1

    od = collections.OrderedDict(sorted(dist_len.items()))
    total_len = sum(od.values())
    print(total_len)
    print(od)
    print([od[od_ele]/total_len for od_ele in od])

    od = collections.OrderedDict(sorted(dist_errs.items()))
    total_errs = sum(od.values())
    print(total_errs)
    print(od)
    print([od[od_ele]/total_errs for od_ele in od])

fd.close()
