
myfilebetter = open('zhen_news_comp_20k_better_comet.txt', 'r')
myfileworse = open('zhen_news_comp_20k_worse_comet.txt', 'r')

cometbetter = open('cometDA_better_comet.txt', 'r')
cometworse = open('cometDA_worse_comet.txt', 'r')

textbetter = open('mqm_better_zhen.txt', 'r')
textworse = open('mqm_worse_zhen.txt', 'r')

scorebetter = open('better-zhen-scores.txt', 'r')
scoreworse = open('worse-zhen-scores.txt', 'r')

id_file = open('zhen-id.txt', 'r')

sys_better = open('better-zhen-sys.txt')
sys_worse = open('worse-zhen-sys.txt')

overlap = 0
comet_non_overlap = 0
my_non_overlap = 0
mynonFile = open('mine-better-than-comet.txt', 'w')
cometnonFile = open('comet-better-than-mine.txt', 'w')
for my_better_score, my_worse_score, c_better_score, c_worse_score, textb, textw, scoreb, scorew, sysB, sysW, id in zip(myfilebetter, myfileworse, cometbetter, cometworse, textbetter, textworse, scorebetter, scoreworse, sys_better, sys_worse, id_file):
    my_better_score, my_worse_score = float(my_better_score[:-1]), float(my_worse_score[:-1])
    c_better_score, c_worse_score = float(c_better_score[:-1]), float(c_worse_score[:-1])

    if my_better_score > my_worse_score and c_better_score > c_worse_score:
        overlap += 1
    elif my_better_score > my_worse_score:
        my_non_overlap += 1
        # write better line one line ahead of worse line
        mynonFile.write(textb[:-1]+f" [Ground Truth: {scoreb[:-1]}, MetaScore: {my_better_score}, Comet: {c_better_score}, id: {id[:-1]}, system: {sysB[:-1]}]"+'\n')
        mynonFile.write(textw[:-1]+f" [Ground Truth: {scorew[:-1]}, MetaScore: {my_worse_score}, Comet: {c_worse_score}, id: {id[:-1]}, system: {sysW[:-1]}]"+'\n')
        mynonFile.write('\n')
    elif c_better_score > c_worse_score:
        comet_non_overlap+=1
        cometnonFile.write(textb[:-1]+f" [Ground Truth: {scoreb[:-1]}, MetaScore: {my_better_score}, Comet: {c_better_score}, id: {id[:-1]}, system: {sysB[:-1]}]"+'\n')
        cometnonFile.write(textw[:-1]+f" [Ground Truth: {scorew[:-1]}, MetaScore: {my_worse_score}, Comet: {c_worse_score}, id: {id[:-1]}, system: {sysW[:-1]}]"+'\n')
        cometnonFile.write('\n')

print(f'my non-overlap: {my_non_overlap}')
print(f'comet non-overlap: {comet_non_overlap}')
print(f'overlap: {overlap}')
