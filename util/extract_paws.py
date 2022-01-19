import csv
import click
import pandas as pd

@click.command()
@click.option('-file')
@click.option('-rl')
def main(file, rl):
    lines = open(file, 'r')
    tsvReader = csv.reader(lines, delimiter="\t")
    header = next(tsvReader)

    df = pd.read_csv('post_final_jan_10_zhen_news_comp_gpu7_num_0.5_del_1.5_mask_1.5_xlm_mbart.csv')
    df = df[["src", "mt", "ref", "score"]]
    df["src"] = df["src"].astype(str)
    df["mt"] = df["mt"].astype(str)
    df["ref"] = df["ref"].astype(str)
    df["score"] = df["score"].astype(float)

    saveCSV = open('both_paws_post_jan10_zhen_news.csv', 'w')
    csvwriter = csv.writer(saveCSV)

    count_0, count_5 = 0, 0
    limit_0 = 25001
    limit_5 = 25001

    if rl:
        fields = ['src', 'mt', 'score']
        csvwriter.writerow(fields)

        for line in tsvReader:
            if int(line[3]) == 1 and count_0 < limit_0:
                count_0 += 1
                csvwriter.writerow([line[1], line[2], 0])
            if int(line[3]) == 0 and count_5 < limit_5:
                count_5 += 1
                csvwriter.writerow([line[1], line[2], -5])

        for mt, ref, score in zip(df["mt"], df["ref"], df["score"]):
            csvwriter.writerow([ref, mt, score])
    else:
        fields = ['src', 'mt', 'ref', 'score']
        csvwriter.writerow(fields)

        for line in tsvReader:
            if int(line[3]) == 1 and count_0 < limit_0:
                count_0 += 1
                csvwriter.writerow([line[1], line[2], 0])
            if int(line[3]) == 0 and count_5 < limit_5:
                count_5 += 1
                csvwriter.writerow([line[1], line[2], -5])

        for mt, ref, score in zip(df["mt"], df["ref"], df["score"]):
            csvwriter.writerow([ref, mt, score])

    print("paws count 0: ", count_0)
    print("paws count -5: ", count_5)

if __name__ == "__main__":
    main()
