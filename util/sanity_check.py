import csv
with open("mqm_newstest2020_zhen.avg_seg_scores.tsv") as score_fd:
    rd = csv.reader(score_fd, delimiter="\t")
    header = next(rd)

    seg_sys_score = {}
    system_name = set()
    for row in rd:
        row = row[0].split()
        seg_sys_score[row[2]+'_'+row[0]] = float(row[1])
        system_name.add(row[0])

    print(system_name)

    num_seg = len(open('newstest2020-ende-ref.de.txt', 'r').readlines())

    ende_file = open(f'mqm_newstest2020_zhen.tsv')
    rd_ende_file = csv.reader(ende_file, delimiter="\t")
    ende_header = next(rd_ende_file)
    rd_ende_dict = {}
    for ende_row in rd_ende_file:
        if ende_row[3]+"<sep>"+ende_row[0] not in rd_ende_dict:
            rd_ende_dict[ende_row[3]+"<sep>"+ende_row[0]] = [ende_row[5], ende_row[6]]

    for i in range(1, num_seg+1):
        for sys in system_name:
            key = str(i) + '<sep>' + sys
            if not key in rd_ende_dict:
                print(key)

# zhen
