import pandas as pd
import click
import csv

@click.command()
@click.option('-file')
def main(file):
    df = pd.read_csv(file)
    df = df[["src", "mt", "ref", "score"]]
    df["src"] = df["src"].astype(str)
    df["mt"] = df["mt"].astype(str)
    df["ref"] = df["ref"].astype(str)
    df["score"] = df["score"].astype(float)

    csvfile = open(f'reference_less_{file}', 'w')
    csvwriter = csv.writer(csvfile)
    fields = ['src', 'mt', 'score']
    csvwriter.writerow(fields)

    for mt, ref, score in zip(df["mt"], df["ref"], df["score"]):
        csvwriter.writerow([ref, mt, score])

    print("Finish reformating to sourceless file!")


if __name__ == "__main__":
    main()
