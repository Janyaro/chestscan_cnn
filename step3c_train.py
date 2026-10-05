
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="chhota test: thori images, 1 epoch")
parser.add_argument("--epochs", type=int, default=6)
args = parser.parse_args()

SPLITS = r"D:\ChestScan-CNN\splits.csv"
CACHE = r"D:\ChestScan-CNN\cache_256.npy"
OUT_DIR = r"D:\ChestScan-CNN"
TAG = "_quick" if args.quick else ""
EPOCHS = 1 if args.quick else args.epochs
BS = 32

torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

df = pd.read_csv(SPLITS)
arr = np.load(CACHE)
y_all = df["label"].values.astype(np.float32)


def get(name, n=None):
    idx = np.where(df["split"].values == name)[0]
    if n and len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    return np.sort(idx)


tr_idx = get("train", 128 if args.quick else None)
va_idx = get("val", 64 if args.quick else None)
te_idx = get("test", 64 if args.quick else None)
print("train/val/test images:", len(tr_idx), len(va_idx), len(te_idx))


def standardize(t):
    """Har image ko apni mean aur spread se barabar karo (brightness ka farq khatam)."""
    m = t.mean(dim=(1, 2), keepdim=True)
    s = t.std(dim=(1, 2), keepdim=True) + 1e-6
    return ((t - m) / s).unsqueeze(1).repeat(1, 3, 1, 1)


def make_train_batch(idx):
    out = []
    for i in idx:
        top, left = np.random.randint(0, 33, size=2)                       # random 224 crop
        im = torch.from_numpy(arr[i, top:top + 224, left:left + 224].astype(np.float32))[None]
        out.append(TF.rotate(im, float(np.random.uniform(-7, 7)))[0])      # halka rotation
    return standardize(torch.stack(out))


def make_eval_batch(idx):
    return standardize(torch.from_numpy(arr[idx, 16:240, 16:240].astype(np.float32)))   # beech ka crop


model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

n_pos = float(y_all[tr_idx].sum())
n_neg = len(tr_idx) - n_pos
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


@torch.no_grad()
def predict(idx, bs=64):
    model.eval()
    out = []
    for k in range(0, len(idx), bs):
        x = make_eval_batch(idx[k:k + bs]).to(device)
        out.append(torch.sigmoid(model(x).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": roc_auc_score(y, p), "sens": tp / max(tp + fn, 1),
            "spec": tn / max(tn + fp, 1), "acc": (tp + tn) / len(y)}, (tn, fp, fn, tp)


best_auc, best_path = -1.0, f"{OUT_DIR}\\best_model{TAG}.pt"
for ep in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    perm = np.random.permutation(tr_idx)
    losses = []
    for k in range(0, len(perm), BS):
        b = perm[k:k + BS]
        if len(b) < 2:
            continue
        x = make_train_batch(b).to(device)
        y = torch.tensor(y_all[b], device=device)
        opt.zero_grad()
        loss = crit(model(x).squeeze(1), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    m, _ = metrics(y_all[va_idx], predict(va_idx))
    print(f"epoch {ep}/{EPOCHS} | loss {np.mean(losses):.3f} | val AUC {m['auc']:.3f} "
          f"sens {m['sens']:.3f} spec {m['spec']:.3f} | {time.time() - t0:.0f} sec")
    if m["auc"] > best_auc:
        best_auc = m["auc"]
        torch.save(model.state_dict(), best_path)

# best epoch (val AUC se chuna gaya) ko test par EK baar parakho
model.load_state_dict(torch.load(best_path, map_location=device))
print("\nBest model (chosen on val AUC):")
for name, idx in [("val", va_idx), ("test", te_idx)]:
    p = predict(idx)
    m, (tn, fp, fn, tp) = metrics(y_all[idx], p)
    print(f"{name}: AUC {m['auc']:.3f} | sensitivity {m['sens']:.3f} | specificity {m['spec']:.3f} "
          f"| accuracy {m['acc']:.3f} | TN={tn} FP={fp} FN={fn} TP={tp}")
    out = df.iloc[idx][["path", "label", "width", "height"]].copy()
    out["prob"] = p
    out.to_csv(f"{OUT_DIR}\\preds_{name}{TAG}.csv", index=False)

print("\nReference: size-only baseline ROC-AUC  val 0.931 | test 0.895")