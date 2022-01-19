import pandas as pd
import csv
import click

@click.command()
@click.option('-type')
@click.option('-file')
def main(type, file):
    df = pd.read_csv(file)
    csvfile = open(f'rescale_{file}', 'w')
    csvwriter = csv.writer(csvfile)
    if type == "rl":
        df = df[["src", "mt", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype(float)

        fields = ['src', 'mt', 'score']
        csvwriter.writerow(fields)

        for src, mt, score in zip(df["src"], df["mt"], df["score"]):
            csvwriter.writerow([src, mt, (25+score)/25])
    else:
        df = df[["src", "mt", "ref", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["ref"] = df["ref"].astype(str)
        df["score"] = df["score"].astype(float)

        fields = ['src', 'mt', 'ref', 'score']
        csvwriter.writerow(fields)

        for src, mt, ref, score in zip(df["src"], df["mt"], df["ref"], df["score"]):
            csvwriter.writerow([src, mt, ref, (25+score)/25])

    print(f"rescale_{file} is generated!")


if __name__ == "__main__":
    main()
