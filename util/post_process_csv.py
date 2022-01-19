import csv
import click

# @click.command()
# @click.option('-rl')
def main():
    file1 = open('jan_11_ende_num_1_del_1.5_mask_1.5_xlm_mbart.csv', 'r')
    File1Reader = csv.reader(file1, delimiter=",")
    header = next(File1Reader)

    lines = open('de_train.tsv', 'r')
    tsvReader = csv.reader(lines, delimiter="\t")
    header = next(tsvReader)

    saveFile = open('post_jan_12_de.csv', 'w')
    csvwriter = csv.writer(saveFile)
    fields = ['src', 'mt', 'score']
    csvwriter.writerow(fields)

    zero_count, n1_count, n2_count, n5_count, n6_count,  n7_count, n8_count, n10_count, n11_count, n12_count, n15_count, n16_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for row in File1Reader:
        if float(row[3]) == 0:
            if zero_count < 3000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                zero_count += 1

        elif float(row[3]) == -1:
            if n1_count < 20000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n1_count += 1

        elif float(row[3]) == -2:
            if n2_count < 10000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n2_count += 1

        elif float(row[3]) == -5:
            if n5_count < 10000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n5_count += 1

        elif float(row[3]) == -6:
            if n6_count < 5000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n6_count += 1

        elif float(row[3]) == -7:
            if n7_count < 5000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n7_count += 1

        elif float(row[3]) == -8:
            if n8_count < 2000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n8_count += 1

        elif float(row[3]) == -10:
            if n10_count < 2000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n10_count += 1

        elif float(row[3]) == -11:
            if n11_count < 1000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n11_count += 1

        elif float(row[3]) == -12:
            if n12_count < 1000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n12_count += 1

        elif float(row[3]) == -15:
            if n15_count < 1000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n15_count += 1

        elif float(row[3]) == -16:
            if n16_count < 1000:
                csvwriter.writerow([row[2], row[1], float(row[3])])
                n16_count += 1

        else:
            csvwriter.writerow([row[2], row[1], float(row[3])])

    print("Finish generation for the first file!")

    limit_0 = 30000

    count_0_diff = 0
    for line in tsvReader:
        if len(line) == 4:
            if int(line[3]) == 1 and count_0_diff < limit_0:
                count_0_diff += 1
                csvwriter.writerow([line[1], line[2], 0])

    print("post_jan_12_de.csv file is processed!")

if __name__ == "__main__":
    main()
