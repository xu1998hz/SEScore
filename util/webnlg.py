import pandas as pd
from itertools import combinations
import math

df = pd.read_csv('all_data_final_averaged.csv')


seg_dict = {}

team_set = set()

semantics_better = open('better_semantics.txt', 'w')
grammar_better = open('better_grammar.txt', 'w')
fluency_better = open('better_fluency.txt', 'w')

semantics_worse = open('worse_semantics.txt', 'w')
grammar_worse = open('worse_grammar.txt', 'w')
fluency_worse = open('worse_fluency.txt', 'w')

ids_ls = open('sample-ids.txt', 'r').readlines()
ids_ls = [ind[:-1] for ind in ids_ls][:-1]

mr_file = open('MRs.txt', 'r').readlines()
mr_ls = [mr_ele[:-1] for mr_ele in mr_file]
#print(mr_file)

ids_ls = [int(ind) for ind in ids_ls]
print(len(ids_ls))

complete_ref = open('gold-sample-reference0.lex', 'r').readlines()

for id, mr, team, text, fluency, grammar, semantics in zip(df['id'], df['mr'], df['team'], df['text'], df['fluency'], df['grammar'], df['semantics']):
    team_set.add(team)

    if mr not in seg_dict:
        seg_dict[mr] = {}

    if team not in seg_dict[mr]:
        seg_dict[mr][team] = {}

    seg_dict[mr][team]['text'] = text
    seg_dict[mr][team]['fluency'] = fluency
    seg_dict[mr][team]['grammar'] = grammar
    seg_dict[mr][team]['semantics'] = semantics
    seg_dict[mr][team]['total'] = float(semantics) + float(fluency) + float(grammar)

print(len(seg_dict))

sys_comb = list(combinations(team_set,2))

gt_semantic_file = open('gt_semantic.txt', 'w')
gt_grammar_file = open('gt_grammar.txt', 'w')
gt_fluency_file = open('gt_fluency.txt', 'w')

overall_better = open('overall_better.txt', 'w')
overall_worse = open('overall_worse.txt', 'w')
gt_overall = open('gt_overall.txt', 'w')

mean_val = 0

for index, mr in enumerate(mr_ls):
    if mr in seg_dict:
        for sysA, sysB in sys_comb:
            if sysA in seg_dict[mr] and sysB in seg_dict[mr]:

                if isinstance(seg_dict[mr][sysA]['text'], str) and isinstance(seg_dict[mr][sysB]['text'], str):
                    # print(seg_dict[mr][sysA]['total'])
                    # print(seg_dict[mr][sysB]['total'])
                    # print('---------------------------------------')
                    if seg_dict[mr][sysA]['total'] > seg_dict[mr][sysB]['total']:
                        gt_overall.write(complete_ref[index])
                        if seg_dict[mr][sysA]['text'][-1] != '\n':
                            overall_better.write(seg_dict[mr][sysA]['text']+'\n')
                        else:
                            overall_better.write(seg_dict[mr][sysA]['text'])

                        if seg_dict[mr][sysB]['text'][-1] != '\n':
                            overall_worse.write(seg_dict[mr][sysB]['text']+'\n')
                        else:
                            overall_worse.write(seg_dict[mr][sysB]['text'])

                    elif seg_dict[mr][sysA]['total'] < seg_dict[mr][sysB]['total']:
                        gt_overall.write(complete_ref[index])
                        if seg_dict[mr][sysB]['text'][-1] != '\n':
                            overall_better.write(seg_dict[mr][sysB]['text']+'\n')
                        else:
                            overall_better.write(seg_dict[mr][sysB]['text'])

                        if seg_dict[mr][sysA]['text'][-1] != '\n':
                            overall_worse.write(seg_dict[mr][sysA]['text']+'\n')
                        else:
                            overall_worse.write(seg_dict[mr][sysA]['text'])

                    if float(seg_dict[mr][sysA]['semantics']) > float(seg_dict[mr][sysB]['semantics']):
                        gt_semantic_file.write(complete_ref[index])

                        if seg_dict[mr][sysA]['text'][-1] != '\n':
                            semantics_better.write(seg_dict[mr][sysA]['text']+'\n')
                        else:
                            semantics_better.write(seg_dict[mr][sysA]['text'])

                        if seg_dict[mr][sysB]['text'][-1] != '\n':
                            semantics_worse.write(seg_dict[mr][sysB]['text']+'\n')
                        else:
                            semantics_worse.write(seg_dict[mr][sysB]['text'])

                    elif float(seg_dict[mr][sysA]['semantics']) < float(seg_dict[mr][sysB]['semantics']):
                        gt_semantic_file.write(complete_ref[index])

                        if seg_dict[mr][sysB]['text'][-1] != '\n':
                            semantics_better.write(seg_dict[mr][sysB]['text']+'\n')
                        else:
                            semantics_better.write(seg_dict[mr][sysB]['text'])

                        if seg_dict[mr][sysA]['text'][-1] != '\n':
                            semantics_worse.write(seg_dict[mr][sysA]['text']+'\n')
                        else:
                            semantics_worse.write(seg_dict[mr][sysA]['text'])

                    if float(seg_dict[mr][sysA]['fluency']) > float(seg_dict[mr][sysB]['fluency']):
                        gt_fluency_file.write(complete_ref[index])

                        if seg_dict[mr][sysA]['text'][-1] != '\n':
                            fluency_better.write(seg_dict[mr][sysA]['text']+'\n')
                        else:
                            fluency_better.write(seg_dict[mr][sysA]['text'])

                        if seg_dict[mr][sysB]['text'][-1] != '\n':
                            fluency_worse.write(seg_dict[mr][sysB]['text']+'\n')
                        else:
                            fluency_worse.write(seg_dict[mr][sysB]['text'])

                    elif float(seg_dict[mr][sysA]['fluency']) < float(seg_dict[mr][sysB]['fluency']):
                        gt_fluency_file.write(complete_ref[index])

                        if seg_dict[mr][sysB]['text'][-1] != '\n':
                            fluency_better.write(seg_dict[mr][sysB]['text']+'\n')
                        else:
                            fluency_better.write(seg_dict[mr][sysB]['text'])

                        if seg_dict[mr][sysA]['text'][-1] != '\n':
                            fluency_worse.write(seg_dict[mr][sysA]['text']+'\n')
                        else:
                            fluency_worse.write(seg_dict[mr][sysA]['text'])

                    if float(seg_dict[mr][sysA]['grammar']) > float(seg_dict[mr][sysB]['grammar']):
                        gt_grammar_file.write(complete_ref[index])

                        if seg_dict[mr][sysA]['text'][-1] != '\n':
                            grammar_better.write(seg_dict[mr][sysA]['text']+'\n')
                        else:
                            grammar_better.write(seg_dict[mr][sysA]['text'])

                        if seg_dict[mr][sysB]['text'][-1] != '\n':
                            grammar_worse.write(seg_dict[mr][sysB]['text']+'\n')
                        else:
                            grammar_worse.write(seg_dict[mr][sysB]['text'])

                    elif float(seg_dict[mr][sysA]['grammar']) < float(seg_dict[mr][sysB]['grammar']):
                        gt_grammar_file.write(complete_ref[index])

                        if seg_dict[mr][sysB]['text'][-1] != '\n':
                            grammar_better.write(seg_dict[mr][sysB]['text']+'\n')
                        else:
                            grammar_better.write(seg_dict[mr][sysB]['text'])

                        if seg_dict[mr][sysA]['text'][-1] != '\n':
                            grammar_worse.write(seg_dict[mr][sysA]['text']+'\n')
                        else:
                            grammar_worse.write(seg_dict[mr][sysA]['text'])

print("All the files are generated!")

#print(seg_dict)
