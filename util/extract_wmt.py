import csv

daFile = open('DArr-seglevel.csv', 'r')
daReader = csv.reader(daFile, delimiter=",")
header = next(daReader)

en_de_file = open('en-de-seg.txt', 'w')
zh_en_file = open('zh-en-seg.txt', 'w')

for row in daReader:
    row_ls = row[0].split()
    lp, id, better, worse, score = row_ls[0], row_ls[2].split("::")[1], row_ls[3], row_ls[4], float(row_ls[5])
    if lp == 'en-de' and score >= 25:
        en_de_file.write(id+','+better+','+worse+'\n')
    elif lp == 'zh-en' and score >= 25:
        zh_en_file.write(id+','+better+','+worse+'\n')

print("Two files are generated!")
