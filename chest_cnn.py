import os
import re
from collections import Counter, defaultdict

DATA = r"D:/ChestScan-CNN/chest_xray"   # wahi folder jahan train, val, test hain
EXT = (".jpeg", ".jpg", ".png")


def list_files(split, cls):
    p = os.path.join(DATA, split, cls)
    return [f for f in os.listdir(p) if f.lower().endswith(EXT) and not f.startswith(".")]


def key(name):
    m = re.match(r"person(\d+)_(bacteria|virus)_", name)
    if m:
        return m.group(2) + ":" + m.group(1)          # jaise bacteria:1000
    m = re.search(r"((?:NORMAL\d*-)?)IM-(\d+)", name)
    if m:
        return "normal:" + m.group(1) + m.group(2)    # jaise normal:0115
    return None


# --- Part 1: kya ek hi number bacteria aur virus dono ke liye istemal hua? ---
nums = defaultdict(set)
for f in list_files("train", "PNEUMONIA"):
    m = re.match(r"person(\d+)_(bacteria|virus)_", f)
    if m:
        nums[m.group(2)].add(m.group(1))
print("Part 1")
print("bacteria numbers:", len(nums["bacteria"]))
print("virus numbers:", len(nums["virus"]))
print("numbers used by BOTH:", len(nums["bacteria"] & nums["virus"]))

# --- Part 2: naya patient key = (qisam, number) ---
keys, where = {}, defaultdict(list)
print("\nPart 2")
for split in ["train", "val", "test"]:
    s = set()
    for cls in ["NORMAL", "PNEUMONIA"]:
        for f in list_files(split, cls):
            k = key(f)
            if k:
                s.add(k)
                where[(split, k)].append(f)
    keys[split] = s
    print(split, "unique keys:", len(s))

both = sorted(keys["train"] & keys["test"])
print("keys in BOTH train and test:", len(both))
print("keys in BOTH train and val:", len(keys["train"] & keys["val"]))
print("keys in BOTH val and test:", len(keys["val"] & keys["test"]))
print("overlap by type:", dict(Counter(k.split(":")[0] for k in both)))

print("\nExamples:")
for k in both[:6]:
    print(k, "| train:", where[("train", k)][:2], "| test:", where[("test", k)][:2])