import numpy as np
import pandas as pd
from PIL import Image

SPLITS = r"D:/ChestScan-CNN\splits.csv"
OUT = r"D:/ChestScan-CNN/cache_256.npy"
GRID = r"D:/ChestScan-CNN/sample_grid.png"
SIZE = 256

df = pd.read_csv(SPLITS)
arr = np.zeros((len(df), SIZE, SIZE), dtype=np.uint8)

for i, p in enumerate(df["path"]):
    im = Image.open(p).convert("L")                       # sab ko grayscale
    w, h = im.size
    s = SIZE / min(w, h)                                  # chhoti side = 256 (shape bigarti nahi)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.Resampling.LANCZOS)
    w, h = im.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2          # beech se 256x256 kaato
    arr[i] = np.asarray(im.crop((left, top, left + SIZE, top + SIZE)))
    if (i + 1) % 500 == 0:
        print(i + 1, "/", len(df))

np.save(OUT, arr)
print("saved:", OUT, arr.shape)

print("\naverage pixel value by split and label (0 = NORMAL, 1 = PNEUMONIA):")
df["mean_px"] = arr.reshape(len(df), -1).mean(axis=1)
print(df.groupby(["split", "label"])["mean_px"].mean().round(1))

# 8 namune ki tasveer (oopar 4 NORMAL, neeche 4 PNEUMONIA)
pick = list(df[df.label == 0].sample(4, random_state=1).index) + list(df[df.label == 1].sample(4, random_state=1).index)
sheet = Image.new("L", (SIZE * 4, SIZE * 2))
for n, idx in enumerate(pick):
    sheet.paste(Image.fromarray(arr[idx]), ((n % 4) * SIZE, (n // 4) * SIZE))
sheet.save(GRID)
print("saved:", GRID)
