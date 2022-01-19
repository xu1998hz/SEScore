import click
import re
import string

@click.command()
@click.option('-filename')
@click.option('-out')
def main(filename, out):
    inFile = open(filename, 'r')
    outFile = open(out, 'w')
    for line in inFile:
        line_str = line[:-1]
        line_str = re.sub('([.,!?""()])', r' \1 ', line_str)
        outFile.write(line_str+'\n')
    print("File is preprocessed and saved!")

if __name__ == "__main__":
    main()
