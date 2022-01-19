import csv

file = open('jan_11_ende_zhen_num_1_del_1.5_mask_1.5_xlm_mbart.tsv', 'r')

lines = open('jan_11_ende_num_1_del_1.5_mask_1.5_xlm_mbart.csv', 'r')
csvReader = csv.reader(lines, delimiter=",")
header = next(csvReader)

limit = 50000
count = 0

saveFile = open('jan_13_ende_news_no_delete.csv', 'w')
csvwriter = csv.writer(saveFile)
fields = ['src', 'mt', 'score']
csvwriter.writerow(fields)

for row, saverow in zip(file, csvReader):
    if count < limit:
        if 'Delete' not in row:
            csvwriter.writerow([saverow[2], saverow[1], float(saverow[3])])
            count += 1
    #     xlm_replace_count+=1

print("Finish!")
