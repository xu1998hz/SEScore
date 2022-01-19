import click

@click.command()
@click.option('-f')
def main(f):
    file = open(f, 'r')
    for line in file:
        if line == None:
            print("None detected")

if __name__ == "__main__":
    main()
