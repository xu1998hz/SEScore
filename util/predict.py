import pandas as pd
import matplotlib.pyplot as plt
from comet import download_model, load_from_checkpoint

df = pd.read_csv('data/zhen-comet.csv')
df = df[["src", "mt", "ref", "score"]]
df["src"] = df["src"].astype(str)
df["mt"] = df["mt"].astype(str)
df["ref"] = df["ref"].astype(str)
df["score"] = df["score"].astype(float)

gt_scores = []
data = []
path = 'lightning_logs/gpu_3456/checkpoints/epoch=7-step=9999.ckpt'
if path == None:
    path = download_model("wmt20-comet-da")
model = load_from_checkpoint(path)

for src, mt, ref, score in zip(df["src"], df["mt"], df["ref"], df["score"]):
    segment = {}
    segment['src'], segment['mt'], segment['ref'] = src, mt, ref
    gt_scores.append(score)
    data.append(segment)

seg_scores, _ = model.predict(data, batch_size=2048, gpus=1)

plt.scatter(gt_scores, seg_scores, marker='.')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.savefig('scatter.png')
print('saved!')
