import csv
import click
from itertools import combinations
import numpy as np

def correct_indices():
    better_file = open("my_results_better_zhen_76064.txt", 'r')
    worse_file = open("my_results_worse_zhen_76064.txt", 'r')

    better_scores = []
    for better_line in better_file:
        score = float(better_line.split("score: ")[1][:-1])
        better_scores.append(score)
    better_scores = np.array(better_scores)

    worse_scores = []
    for worse_line in worse_file:
        score = float(worse_line.split("score: ")[1][:-1])
        worse_scores.append(score)
    worse_scores = np.array(worse_scores)

    total = better_scores.shape[0]
    correct = better_scores <= worse_scores
    return correct

@click.command()
@click.option('-dir')
def main(dir):
    sentence_fd = open(f"mqm_newstest2020_{dir}.tsv", 'r')
    rd = csv.reader(sentence_fd, delimiter="\t")
    header = next(rd)

    id_sys_err_dict = {}
    for row in rd:
        key, val = row[3]+' '+row[0], row[-1]+' '+row[-2]
        if key in id_sys_err_dict:
            id_sys_err_dict[key].add(val)
        else:
            id_sys_err_dict[key]=set()
            id_sys_err_dict[key].add(val)
    #print(len(id_sys_err_dict))

    score_fd = open(f"mqm_newstest2020_{dir}.avg_seg_scores.tsv", 'r')
    rd = csv.reader(score_fd, delimiter="\t")
    header = next(rd)

    system_name = set()
    for row in rd:
        row = row[0].split()
        system_name.add(row[0])

    num_seg = len(open(f'newstest2020-{dir}-ref.{dir[-2:]}.txt', 'r').readlines())
    sys_comb = list(combinations(system_name,2))

    index_id_sys_dict = {}
    index = 0
    for i in range(1, num_seg+1):
        for sys_pair in sys_comb:
            sysA, sysB = sys_pair
            index_id_sys_dict[index] = (str(i), sysA, sysB)
            index+=1

    correct_arr = correct_indices()
    err_types = []
    for index, truth in enumerate(correct_arr):
        if truth:
            id, sysA, sysB = index_id_sys_dict[index]
            if id+' '+sysA in id_sys_err_dict and id+' '+sysB in id_sys_err_dict:
                err_types.extend(list(id_sys_err_dict[id+' '+sysA]))
                err_types.extend(list(id_sys_err_dict[id+' '+sysB]))

    err_types_dict = {}
    for err_type in err_types:
        if err_type in err_types_dict:
            err_types_dict[err_type] += 1
        else:
             err_types_dict[err_type] = 1

    subtotal = 0
    total = sum(err_types_dict.values())
    for key, val in err_types_dict.items():
        if val > 1000:
            subtotal+=val
            print(key, val)
    print('--------------------------------------')
    print("Displayed number of errors: ", subtotal)
    print("Total number of errors: ", total)

if __name__ == "__main__":
    main()
