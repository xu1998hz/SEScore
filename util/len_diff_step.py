import pandas as pd
import csv
import click

@click.command()
@click.option('-file')
def main(file):
    df = pd.read_csv(file)
    df = df[["src", "mt", "ref", "score"]]
    df["src"] = df["src"].astype(str)
    df["mt"] = df["mt"].astype(str)
    df["ref"] = df["ref"].astype(str)
    df["score"] = df["score"].astype(float)

    # refactorFile = open('post_rm_'+file, 'w')
    # csvwriter = csv.writer(refactorFile)
    # fields = ["src", "mt", "ref", "score"]
    # csvwriter.writerow(fields)

    for src, mt, ref, score in zip(df["src"], df["mt"], df["ref"], df["score"]):
        if len(mt.split()) - len(ref.split()) < -10:
            #print(len(ref.split()))
            if len(ref.split()) < 50:
                print(mt)
                print(ref)
                print('-----------------------------------')
            #Ecsvwriter.writerow([src, mt, ref, score])
    #
    # print("remove some severe error data!")

if __name__ == "__main__":
    main()
