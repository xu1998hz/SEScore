import click
import random

@click.command()
@click.option('-f1')
@click.option('-f2')
@click.option('-lang')
def main(f1, f2, lang):
    file1, file2 = open(f1, 'r'), open(f2, 'r')
    lines_1, lines_2 = set(), set()
    newlines_1, newlines_2 = [], []

    for line_1, line_2 in zip(file1, file2):
        if line_1 != '\n' and line_2 != '\n':
            len_1_old, len_2_old = len(lines_1), len(lines_2)
            lines_1.add(line_1)
            lines_2.add(line_2)
            len_1_new, len_2_new = len(lines_1), len(lines_2)

            if (len_1_new-len_1_old)==1 and (len_2_new-len_2_old)==1:
                newlines_1.append(line_1)
                newlines_2.append(line_2)

    print(len(lines_1))
    print(len(lines_2))

    print(len(newlines_1))
    print(len(newlines_2))

    selected_sens = sorted(random.sample(range(len(newlines_1)), 40000))
    refFile = open('train.ref.'+lang, 'w')
    srcFile = open('train.src.'+lang, 'w')


    for index in selected_sens:
        refFile.write(newlines_1[index])
        srcFile.write(newlines_2[index])

    print("Two files are generated!")

if __name__ == "__main__":
    main()
