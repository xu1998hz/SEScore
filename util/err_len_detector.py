import pandas as pd
import click

@click.command()
@click.option('-file')
def main(file):
    file_name = open(file, 'r')

    df = pd.read_csv(file_name)
    df = df[["src", "mt", "ref", "score"]]
    df["src"] = df["src"].astype(str)
    df["mt"] = df["mt"].astype(str)
    df["ref"] = df["ref"].astype(str)
    df["score"] = df["score"].astype(float)

    count = 0
    total = 0
    for mt, ref in zip(df["mt"], df["ref"]):
        if len(mt.split()) - len(ref.split()) < -10:
            count+=1
            print(mt)
            print(ref)
            print()
        total+=1

    print(count)
    print(total)

if __name__ == "__main__":
    main()
