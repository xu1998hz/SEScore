import csv
import re

def score_retrieve(system, id, mqmScore_dict):
    key = id+'_'+system
    if key in mqmScore_dict:
        return mqmScore_dict[key]
    return None

def sentence_retrieve(system, id, mqm_dict):
    key = id+'_'+system
    if key in mqm_dict:
        return mqm_dict[key]
    return None

mqmScoreFile = open('mqm_newstest2020_ende.avg_seg_scores.tsv', 'r')
mqmScoreReader = csv.reader(mqmScoreFile, delimiter="\t")
header = next(mqmScoreReader)

mqmScore_dict = {}
for row in mqmScoreFile:
    row = row.split()
    msystem, mscore, mid = row[0], row[1], row[2]
    mqmScore_dict[mid+"_"+msystem] = float(mscore)

wmtScoreFile = open('en-de-seg.txt', 'r')

mqmFile = open('mqm_newstest2020_ende.tsv', 'r')
mqmReader = csv.reader(mqmFile, delimiter="\t")
header = next(mqmReader)

mqm_dict = {}
for row in mqmFile:
    row = row.split('\t')
    CLEANR = re.compile('<.*?>')
    print(row)
    msystem, msentence, mid = row[0], row[6], row[3]
    cleantext = re.sub(CLEANR, '', msentence)
    mqm_dict[mid+'_'+msystem] = cleantext

betterFile = open('mqm_better_ende.txt', 'w')
worseFile = open('mqm_worse_ende.txt', 'w')
iter = 0

for line in wmtScoreFile:
    seg_id, systemA, systemB = line.split(',')[0], line.split(',')[1], line.split(',')[2][:-1]
    #print(systemB)
    scoreA = score_retrieve(systemA, seg_id, mqmScore_dict)
    scoreB = score_retrieve(systemB, seg_id, mqmScore_dict)
    if scoreA and scoreB:
        iter += 1
        sentenceA = sentence_retrieve(systemA, seg_id, mqm_dict)
        sentenceB = sentence_retrieve(systemB, seg_id, mqm_dict)
        if sentenceA and sentenceB:
            if scoreA > scoreB:
                betterFile.write(sentenceA+'\n')
                worseFile.write(sentenceB+'\n')
            elif scoreB > scoreA:
                betterFile.write(sentenceB+'\n')
                worseFile.write(sentenceA+'\n')
        else:
            print(sentenceA)
            print(sentenceB)

print(iter)
print("Two files are generated!")
