import pandas as pd
import click

@click.command()
@click.option('-file')
def main(file):
    df = pd.read_csv(file, header=0, index_col=False)
    ds = df.sample(frac=1)
    ds.to_csv('shuffle_'+file, index=False)

    print("File is shuffled")

if __name__ == "__main__":
    main()
