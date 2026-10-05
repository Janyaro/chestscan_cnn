"""Shared inference code: preprocessing identical to training, prediction and Grad-CAM."""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

SIZE, CROP, MARGIN = 256, 224, 16


def load_model(path, device="cpu"):
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def to_cache_array(img):
    """PIL image -> 256x256 uint8 array, same steps as the training cache."""
    im = img.convert("L")
    w, h = im.size
    s = SIZE / min(w, h)                      # shorter side -> 256, shape is not distorted
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.Resampling.LANCZOS)
    w, h = im.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    return np.asarray(im.crop((left, top, left + SIZE, top + SIZE)))


def to_tensor(arr256):
    """256x256 uint8 -> (1, 3, 224, 224), standardised per image like in training."""
    x = torch.from_numpy(arr256[MARGIN:MARGIN + CROP, MARGIN:MARGIN + CROP].astype(np.float32))
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x[None, None].repeat(1, 3, 1, 1)


def predict_with_cam(model, arr256):
    """Returns (probability of pneumonia, Grad-CAM map in [0, 1], the 224x224 crop shown to the model)."""
    x = to_tensor(arr256)
    store = {}

    def fwd(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = model.layer4.register_forward_hook(fwd)
    model.zero_grad()
    logit = model(x).squeeze()
    logit.backward()
    handle.remove()
    a, g = store["a"][0], store["g"][0]
    cam = torch.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = torch.nn.functional.interpolate(cam[None, None], size=(CROP, CROP),
                                          mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = (cam / (cam.max() + 1e-8)).detach().numpy()
    return torch.sigmoid(logit).item(), cam, arr256[MARGIN:MARGIN + CROP, MARGIN:MARGIN + CROP]


def overlay(crop, cam, alpha=0.5):
    """Grayscale crop + heat map. Where the map is 0 the picture stays plain (no evidence)."""
    import matplotlib
    base = np.repeat((crop.astype(np.float32) / 255.0)[..., None], 3, axis=2)
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    a = alpha * cam[..., None]
    return np.clip((1 - a) * base + a * heat, 0, 1)
