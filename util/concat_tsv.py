import csv
import glob

saveFile = open('exp_stepwise_score_zhen/news_complementary_zhen_200k.csv', 'w')
csvwriter = csv.writer(saveFile)
fields = ['src', 'mt', 'ref', 'score']
csvwriter.writerow(fields)

for subFile in glob.glob('exp_stepwise_score_zhen/news_complementary_zhen/*'):
    sentence_fd = open(subFile, 'r')
    print(subFile)
    rd = csv.reader(sentence_fd, delimiter=",")
    print(rd)
    header = next(rd)

    for row in rd:
        #print(row)
        assert(len(row) == 4)
        csvwriter.writerow([row[0], row[1], row[2], float(row[3])])

print("All files are concatenated!")
