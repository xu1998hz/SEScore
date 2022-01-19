import click
import os

"""Yield batch sized list of sentences."""
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

@click.command()
@click.option('-index_range')
@click.option('-f')
@click.option('-folder')
@click.option('-sdir')
@click.option('-rdir')
def main(index_range, f, folder, sdir, rdir):
    lines = open(f, 'r').readlines()
    if not os.path.exists(folder):
        os.makedirs(folder)
    if sdir:
        if not os.path.exists(folder+'/'+sdir):
            os.makedirs(folder+'/'+sdir)
    else:
        if not os.path.exists(folder+'/'+rdir):
            os.makedirs(folder+'/'+rdir)
    # dynamic generate subfile
    if sdir:
        for index, sub_lines in enumerate(batchify(lines, int(index_range))):
            saveFile = open(folder+'/'+sdir+'/'+str(index)+'_'+f, 'w')
            saveFile.writelines(sub_lines)
    else:
        for index, sub_lines in enumerate(batchify(lines, int(index_range))):
            saveFile = open(folder+'/'+rdir+'/'+str(index)+'_'+f, 'w')
            saveFile.writelines(sub_lines)

    print(f'All files are generated in {folder}!')

if __name__ == "__main__":
    main()
