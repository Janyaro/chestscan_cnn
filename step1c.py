import os
import random

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA = r"D:/ChestScan-CNN/chest_xray"   # wahi folder jahan train, val, test hain
EXT = (".jpeg", ".jpg", ".png")
N = 300                                  # har class se kitni images lein
random.seed(0)

rows, labels = [], []
for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
    folder = os.path.join(DATA, "train", cls)
    files = [f for f in os.listdir(folder) if f.lower().endswith(EXT) and not f.startswith(".")]
    for f in random.sample(files, min(N, len(files))):
        im = Image.open(os.path.join(folder, f)).convert("L")
        w, h = im.size
        a = np.asarray(im.resize((64, 64)), dtype=float)
        rows.append([w, h, w / h, a.mean(), a.std()])
        labels.append(label)

X, y = np.array(rows), np.array(labels)

print("Class-wise averages (width, height, aspect ratio, brightness, brightness spread)")
for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
    m = X[y == label].mean(axis=0)
    print(f"{cls:10s} width={m[0]:.0f} height={m[1]:.0f} aspect={m[2]:.2f} "
          f"brightness={m[3]:.1f} spread={m[4]:.1f}")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
auc = cross_val_score(GradientBoostingClassifier(random_state=0), X, y, cv=cv, scoring="roc_auc")
print(f"\nROC-AUC from ONLY size and brightness: {auc.mean():.3f} (+/- {auc.std():.3f})")