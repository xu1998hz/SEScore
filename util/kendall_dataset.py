import csv
from itertools import combinations
import re
import click

@click.command()
@click.option('-dir')
def main(dir):
    score_fd = open(f"mqm_newstest2020_{dir}.avg_seg_scores.tsv", 'r')
    rd = csv.reader(score_fd, delimiter="\t")
    header = next(rd)

    seg_sys_score = {}
    system_name = set()
    for row in rd:
        row = row[0].split()
        seg_sys_score[row[2]+'_'+row[0]] = float(row[1])
        system_name.add(row[0])

    sentence_fd = open(f"mqm_newstest2020_{dir}.tsv", 'r')
    rd = csv.reader(sentence_fd, delimiter="\t")
    header = next(rd)

    seg_sys_sentence = {}
    CLEANR = re.compile('<.*?>')
    for row in rd:
        cleantext = re.sub(CLEANR, '', row[6])
        if not '\n' in cleantext:
            seg_sys_sentence[row[3]+'_'+row[0]] = cleantext
        if row[3] == '181':
            print(row)

    num_seg = len(open(f'newstest2020-{dir}-ref.{dir[-2:]}.txt', 'r').readlines())
    sys_comb = list(combinations(system_name,2))

    betterFile = open(f'mqm_better_{dir}.txt', 'w')
    worseFile = open(f'mqm_worse_{dir}.txt', 'w')

    inSrcLines = open(f'newstest2020-{dir}-src.{dir[:2]}.txt', 'r').readlines()
    inRefLines = open(f'newstest2020-{dir}-ref.{dir[-2:]}.txt', 'r').readlines()

    outSrcFile = open(f'eval-{dir}-src.txt', 'w')
    outRefFile = open(f'eval-{dir}-ref.txt', 'w')

    better_score = open(f'better-{dir}-scores.txt', 'w')
    worse_score = open(f'worse-{dir}-scores.txt', 'w')

    id_file = open(f'{dir}-id.txt', 'w')

    better_sys = open(f'better-{dir}-sys.txt', 'w')
    worse_sys = open(f'worse-{dir}-sys.txt', 'w')

    iter=0
    err_sens = 0
    equal = 0
    for i in range(1, num_seg+1):
        for sys_pair in sys_comb:
            sysA, sysB = sys_pair
            #print(seg_sys_sentence[str(i)+'_'+sysA])
            if seg_sys_score[str(i)+'_'+sysA] > seg_sys_score[str(i)+'_'+sysB]:
                if str(i)+'_'+sysB in seg_sys_sentence and str(i)+'_'+sysA in seg_sys_sentence:
                    iter+=1
                    betterFile.write(seg_sys_sentence[str(i)+'_'+sysA]+'\n')
                    worseFile.write(seg_sys_sentence[str(i)+'_'+sysB]+'\n')
                    id_file.write(str(i)+'\n')
                    better_sys.write(sysA+'\n')
                    worse_sys.write(sysB+'\n')
                    outSrcFile.write(inSrcLines[i-1])
                    outRefFile.write(inRefLines[i-1])
                    better_score.write(str(seg_sys_score[str(i)+'_'+sysA])+'\n')
                    worse_score.write(str(seg_sys_score[str(i)+'_'+sysB])+'\n')
                else:
                    # print(">")
                    # print(str(i)+'_'+sysA)
                    # print(str(i)+'_'+sysB)
                    # print('-----------------------------------------------------')
                    err_sens+=1
            elif seg_sys_score[str(i)+'_'+sysA] < seg_sys_score[str(i)+'_'+sysB]:
                if str(i)+'_'+sysB in seg_sys_sentence and str(i)+'_'+sysA in seg_sys_sentence:
                    iter+=1
                    betterFile.write(seg_sys_sentence[str(i)+'_'+sysB]+'\n')
                    worseFile.write(seg_sys_sentence[str(i)+'_'+sysA]+'\n')
                    id_file.write(str(i)+'\n')
                    better_sys.write(sysB+'\n')
                    worse_sys.write(sysA+'\n')
                    outSrcFile.write(inSrcLines[i-1])
                    outRefFile.write(inRefLines[i-1])
                    better_score.write(str(seg_sys_score[str(i)+'_'+sysB])+'\n')
                    worse_score.write(str(seg_sys_score[str(i)+'_'+sysA])+'\n')
                else:
                    # print("<")
                    # print(str(i)+'_'+sysA)
                    # print(str(i)+'_'+sysB)
                    # print('-----------------------------------------------------')
                    err_sens+=1
            else:
                equal += 1

    print(iter)
    print(equal)
    print("Err Sens: ", err_sens)
    print("Six files are generated!")

if __name__ == "__main__":
    main()
