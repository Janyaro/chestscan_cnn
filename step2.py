import hashlib
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

DATA = r"D:/ChestScan-CNN/chest_xray"        # wahi folder jahan train, val, test hain
OUT = r"D:/ChestScan-CNN/splits.csv"         # ye file ban jayegi (agle steps isi ko padhenge)
EXT = (".jpeg", ".jpg", ".png")


def key(name):
    m = re.match(r"person(\d+)_(bacteria|virus)_", name)
    if m:
        return m.group(2) + ":" + m.group(1)
    m = re.search(r"((?:NORMAL\d*-)?)IM-(\d+)", name)
    if m:
        return "normal:" + m.group(1) + m.group(2)
    return None


# 1. saari images ki table banao
rows = []
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(DATA, split, cls)
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith(EXT) or f.startswith("."):
                continue
            path = os.path.join(folder, f)
            with open(path, "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            w, h = Image.open(path).size
            rows.append(dict(path=path, orig_split=split, label=int(cls == "PNEUMONIA"),
                             key=key(f), md5=md5, width=w, height=h))
df = pd.DataFrame(rows)
df["key"] = df["key"].fillna(df["path"])      # naam se key na bane to image ka apna group
print("total images:", len(df), "| label 0 = NORMAL, 1 = PNEUMONIA")

# 2. duplicates (train/val se hatao, test ko mat chheero)
dup = df.duplicated("md5", keep="first")
print("duplicate images by original split:", df[dup].orig_split.value_counts().to_dict())
df = df[~(dup & (df.orig_split != "test"))].copy()

# 3. original train + val ko mareez ke hisaab se naye train/val mein baanto
pool = df[df.orig_split != "test"].reset_index(drop=True)
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
_, va_idx = next(sgkf.split(pool, pool["label"], groups=pool["key"]))
pool["split"] = "train"
pool.loc[va_idx, "split"] = "val"
test = df[df.orig_split == "test"].copy()
test["split"] = "test"
final = pd.concat([pool, test], ignore_index=True)

# 4. checks
print("\nimages per split and class:")
print(pd.crosstab(final["split"], final["label"], margins=True))
print("\npatients (keys) per split:", final.groupby("split")["key"].nunique().to_dict())
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    shared = set(final[final.split == a].key) & set(final[final.split == b].key)
    print(f"patients shared by {a} and {b}: {len(shared)}")

# 5. sirf size se andaza (shortcut ka baseline)
def feats(d):
    return np.c_[d.width, d.height, d.width / d.height]

tr, va, te = (final[final.split == s] for s in ["train", "val", "test"])
m = GradientBoostingClassifier(random_state=0).fit(feats(tr), tr["label"])
for name, d in [("val", va), ("test", te)]:
    auc = roc_auc_score(d["label"], m.predict_proba(feats(d))[:, 1])
    print(f"size-only baseline ROC-AUC on {name}: {auc:.3f}")

final.drop(columns=["md5"]).to_csv(OUT, index=False)
print("\nsaved:", OUT)