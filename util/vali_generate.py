import csv
import re
import click

@click.command()
@click.option('-dir')
def main(dir):
    with open(f"mqm_newstest2020_{dir}.avg_seg_scores.tsv") as score_fd:
        rd = csv.reader(score_fd, delimiter="\t")
        header = next(rd)

        ende_file = open(f'mqm_newstest2020_{dir}.tsv')
        rd_ende_file = csv.reader(ende_file, delimiter="\t")
        ende_header = next(rd_ende_file)

        CLEANR = re.compile('<.*?>')

        # load in target matches with src
        tar_lines = open(f'newstest2020-{dir}-ref.{dir[-2:]}.txt', 'r').readlines()
        tar_lines = [line[:-1] for line in tar_lines]

        # load in src and hypothesis here
        rd_ende_dict = {}
        for ende_row in rd_ende_file:
            if ende_row[3]+"<sep>"+ende_row[0] not in rd_ende_dict:
                rd_ende_dict[ende_row[3]+"<sep>"+ende_row[0]] = [ende_row[5], ende_row[6]]

        print(len(rd_ende_dict))

        gt_file = open(f'{dir}-comet.csv', 'w')
        csvwriter = csv.writer(gt_file)
        fields = ['src', 'mt', 'ref', 'score']
        csvwriter.writerow(fields)

        # load in score match with seg id
        for row in rd:
            system, score, seg_id = row[0].split()[0], float(row[0].split()[1]), row[0].split()[2]
            if seg_id+'<sep>'+system in rd_ende_dict:
                cleantext = re.sub(CLEANR, '', rd_ende_dict[seg_id+'<sep>'+system][1])
                csvwriter.writerow([rd_ende_dict[seg_id+'<sep>'+system][0], cleantext, tar_lines[int(seg_id)-1], score])

        print("File is generated! All information is extracted!")

if __name__ == "__main__":
    main()
