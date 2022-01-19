import csv

src_file = open('eval-ende-src.txt', 'r')
ref_file = open('eval-ende-ref.txt', 'r')
mqm_better = open('mqm_better_ende.txt', 'r')
mqm_worse = open('mqm_worse_ende.txt', 'r')

rank_data = open("rank_ende_test.csv", 'w')
csvwriter = csv.writer(rank_data)
fields = ["src", "pos", "neg", "ref"]
csvwriter.writerow(fields)

for src_line, better_line, worse_line, ref_line in zip(src_file, mqm_better, mqm_worse, ref_file):
     csvwriter.writerow([src_line[:-1], better_line[:-1], worse_line[:-1], ref_line[:-1]])

print('rank test file is generated!')
