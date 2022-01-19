from comet import download_model, load_from_checkpoint
import os
import numpy as np
import click
import glob

@click.command()
@click.option('-path')
@click.option('-save')
@click.option('-rl')
def main(path, save, rl):
    srcFile = open('eval-zhen-src.txt', 'r').readlines()
    refFile = open('eval-zhen-ref.txt', 'r').readlines()
    betterFile = open('mqm_better_zhen.txt', 'r').readlines()
    worseFile = open('mqm_worse_zhen.txt', 'r').readlines()
    saveBetter= open(f'{save}_better.txt', 'w')
    saveWorse = open(f'{save}_worse.txt', 'w')
    #for ckpt in glob.glob(path+'*'):
    #model_path = download_model("wmt20-comet-da")
    if path == None:
        path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(path)
    better_data = []
    for src_line, ref_line, better_line in zip(srcFile, refFile, betterFile):
        segment = {}
        if rl:
            segment['src'], segment['mt'] = ref_line, better_line
        else:
            segment['src'], segment['mt'], segment['ref'] = src_line, better_line, ref_line
        better_data.append(segment)

    better_seg_scores, _ = model.predict(better_data, batch_size=512, gpus=1)
    better_seg_scores = np.array(better_seg_scores)

    for better_score in better_seg_scores:
        saveBetter.write(str(better_score)+'\n')

    worse_data = []
    for src_line, ref_line, worse_line in zip(srcFile, refFile, worseFile):
        segment = {}
        if rl:
            segment['src'], segment['mt'] = ref_line, worse_line
        else:
            segment['src'], segment['mt'], segment['ref'] = src_line, worse_line, ref_line
        worse_data.append(segment)

    worse_seg_scores, _ = model.predict(worse_data, batch_size=512, gpus=1)
    worse_seg_scores = np.array(worse_seg_scores)

    for worse_score in worse_seg_scores:
        saveWorse.write(str(worse_score)+'\n')

    print("Two files are saved!")
    assert better_seg_scores.shape[0] == worse_seg_scores.shape[0]

    total = better_seg_scores.shape[0]
    correct = np.sum(better_seg_scores > worse_seg_scores)
    print(correct)
    incorrect = total - correct
    print((correct - incorrect)/total)
    print()

if __name__ == "__main__":
    main()
