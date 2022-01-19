from comet import download_model, load_from_checkpoint
import os
import numpy as np
import click
import glob
import pandas as pd
from scipy import stats

@click.command()
@click.option('-path')
@click.option('-save')
@click.option('-rl')
def main(path, save, rl):
    df = pd.read_csv('data/zhen-comet.csv', index_col=False)
    df = df[["src", "ref", "mt", "score"]]
    src_ls = df["src"].astype(str).tolist()
    ref_ls = df["ref"].astype(str).tolist()
    mt_ls = df["mt"].astype(str).tolist()
    scores_ls = df["score"].astype(float).tolist()
    #for ckpt in glob.glob(path+'*'):
    #model_path = download_model("wmt20-comet-da")
    if path == None:
        path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(path)

    data_ls = []
    for src_line, ref_line, mt_line in zip(src_ls, ref_ls, mt_ls):
        segment = {}
        if rl:
            segment['src'], segment['mt'] = ref_line, mt_line
        else:
            segment['src'], segment['mt'], segment['ref'] = src_line, mt_line, ref_line
        data_ls.append(segment)

    seg_scores, _ = model.predict(data_ls, batch_size=512, gpus=1)
    seg_scores = np.array(seg_scores)
    scores_arr = np.array(scores_ls)

    print(stats.pearsonr(seg_scores, scores_arr))
    print(stats.spearmanr(seg_scores, scores_arr))

if __name__ == "__main__":
    main()
