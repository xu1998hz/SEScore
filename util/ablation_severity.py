import csv

tsvFile = open('jan_11_ende_zhen_num_1_del_1.5_mask_1.5_xlm_mbart.tsv', 'r')
tsvReader = csv.reader(tsvFile, delimiter="\t")

lines = open('jan_11_ende_num_1_del_1.5_mask_1.5_xlm_mbart.csv', 'r')
csvReader = csv.reader(lines, delimiter=",")
header = next(csvReader)

limit = 20000
count = 0

saveFile = open('jan_13_ende_news_severity.csv', 'w')
csvwriter = csv.writer(saveFile)
fields = ['src', 'mt', 'score']
csvwriter.writerow(fields)

count = 0
for row, saverow in zip(tsvReader, csvReader):
    if float(saverow[3]) < -5 and count < limit:
        csvwriter.writerow([saverow[2], saverow[1], -(len(row)-1)])
        count += 1

print(count)
print("Finish!")
