"""Data pipeline for U-Net on Cell Tracking Challenge data.

Implements the paper's training ingredients:
  * 1-px separation borders between touching cells (Sec. 1 / Fig. 3c)
  * pixel-wise weight map  w = w_c + w0 * exp(-(d1+d2)^2 / (2 sigma^2))   (Eq. 2)
  * random elastic deformation: 3x3 coarse grid, displacements ~ N(0, 10 px),
    bicubic interpolation (Sec. 3.1), plus flips/rot90 and gray-value jitter
  * overlap-tile inference with mirror extrapolation (Fig. 2)
  * the challenge's SEG (Jaccard) measure used for "IOU" in Table 2
"""
import glob
import math
import os
import random
import re
import urllib.request
import zipfile

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

CTC_URL = "http://data.celltrackingchallenge.net/training-datasets/{}.zip"


# ----------------------------------------------------------------- loading
def download_ctc(name, root):
    out = os.path.join(root, name)
    if os.path.isdir(out):
        return out
    os.makedirs(root, exist_ok=True)
    zp = os.path.join(root, name + ".zip")
    print("downloading", CTC_URL.format(name))
    urllib.request.urlretrieve(CTC_URL.format(name), zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(root)
    return out


def normalize(img):
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def load_ctc(path, seqs=("01", "02")):
    """Load gold-truth (man_seg) annotated frames: list of dict(name, img, lab)."""
    import tifffile
    items = []
    for s in seqs:
        for f in sorted(glob.glob(os.path.join(path, f"{s}_GT", "SEG", "man_seg*.tif"))):
            idx = re.findall(r"(\d+)", os.path.basename(f))[-1]
            imgf = os.path.join(path, s, f"t{idx}.tif")
            if not os.path.exists(imgf):
                continue
            img = tifffile.imread(imgf).astype(np.float32)
            lab = tifffile.imread(f).astype(np.int32)
            items.append(dict(name=f"{s}_{idx}", img=normalize(img), lab=lab))
    return items


# ----------------------------------------------------------------- targets
def make_targets(lab, w0=10.0, sigma=5.0):
    """Binary mask with 1-px borders between touching cells + weight map (Eq. 2)."""
    lab = lab.astype(np.int32)
    mx = ndi.maximum_filter(lab, size=3)
    border = (lab > 0) & (mx > lab)         # pixel next to a different, higher-id cell
    lab2 = lab.copy()
    lab2[border] = 0
    mask = (lab2 > 0).astype(np.int64)

    # w_c: balance class frequencies (expected weight ~1)
    p = float(np.clip(mask.mean(), 1e-3, 1 - 1e-3))
    wc = np.where(mask == 1, 0.5 / p, 0.5 / (1 - p)).astype(np.float32)

    ids = np.unique(lab2)
    ids = ids[ids > 0]
    if len(ids) >= 2:
        d = np.stack([ndi.distance_transform_edt(lab2 != i) for i in ids]).astype(np.float32)
        d = np.partition(d, 1, axis=0)[:2]
        d1, d2 = d.min(0), d.max(0)
        w = wc + w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
    else:
        w = wc
    return mask, w.astype(np.float32)


# ----------------------------------------------------------------- augmentation
def elastic_deform(img, mask, wgt, sigma=10.0, grid=3):
    """Same random smooth deformation for image (bicubic), weights (bilinear), mask (nearest)."""
    dev = img.device
    _, _, H, W = img.shape
    disp = torch.randn(1, 2, grid, grid, device=dev) * sigma
    disp = F.interpolate(disp, size=(H, W), mode="bicubic", align_corners=True)
    flow = disp.permute(0, 2, 3, 1).clone()
    flow[..., 0] *= 2.0 / (W - 1)
    flow[..., 1] *= 2.0 / (H - 1)
    theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32, device=dev)
    g = F.affine_grid(theta, (1, 1, H, W), align_corners=True) + flow
    kw = dict(padding_mode="reflection", align_corners=True)
    img = F.grid_sample(img, g, mode="bicubic", **kw)
    wgt = F.grid_sample(wgt, g, mode="bilinear", **kw)
    mask = F.grid_sample(mask, g, mode="nearest", **kw)
    return img, mask, wgt


class TileSampler:
    """Samples (input tile 572x572, target 388x388, weights 388x388) from mirror-padded images."""

    def __init__(self, items, tile=572, out=388, w0=10.0, sigma=5.0):
        self.tile, self.out = tile, out
        self.pad = (tile - out) // 2
        self.data = []
        for it in items:
            mask, w = make_targets(it["lab"], w0, sigma)
            p = self.pad
            self.data.append((np.pad(it["img"], p, "reflect"),
                              np.pad(mask, p, "reflect").astype(np.float32),
                              np.pad(w, p, "reflect")))

    def sample(self, device, elastic=True, use_weight=True):
        T, c = self.tile, self.pad
        img, mask, w = random.choice(self.data)
        H, W = img.shape
        y, x = random.randint(0, H - T), random.randint(0, W - T)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[y:y + T, x:x + T]))[None, None].to(device)
        img, mask, w = t(img), t(mask), t(w)
        if elastic:
            img, mask, w = elastic_deform(img, mask, w)
        k = random.randint(0, 3)
        flip = random.random() < 0.5
        def geo(a):
            a = torch.rot90(a, k, (2, 3))
            return torch.flip(a, (3,)) if flip else a
        img, mask, w = geo(img), geo(mask), geo(w)
        img = img * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)   # gray-value variation
        mask = mask[:, 0, c:-c, c:-c].round().long()
        w = w[:, 0, c:-c, c:-c]
        if not use_weight:
            w = torch.ones_like(w)
        return img, mask, w


# ----------------------------------------------------------------- inference
@torch.no_grad()
def predict_full(model, img, device, tile=572, out=388, amp=True, tta=False):
    """Overlap-tile strategy: mirror-extrapolate borders, stitch 388x388 outputs."""
    model.eval()
    H, W = img.shape
    pad = (tile - out) // 2
    nH, nW = math.ceil(H / out), math.ceil(W / out)
    padded = np.pad(img, ((pad, nH * out + pad - H), (pad, nW * out + pad - W)), mode="reflect")
    prob = np.zeros((nH * out, nW * out), np.float32)
    variants = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    for i in range(nH):
        for j in range(nW):
            t = torch.from_numpy(padded[i * out:i * out + tile, j * out:j * out + tile].copy())[None, None].to(device)
            acc = 0
            for k, f in variants:
                x = torch.rot90(t, k, (2, 3))
                x = torch.flip(x, (3,)) if f else x
                with torch.autocast(device_type=device.type, enabled=amp):
                    p = torch.softmax(model(x).float(), 1)[:, 1:2]
                p = torch.flip(p, (3,)) if f else p
                acc = acc + torch.rot90(p, -k, (2, 3))
            prob[i * out:(i + 1) * out, j * out:(j + 1) * out] = (acc / len(variants))[0, 0].cpu().numpy()
    return prob[:H, :W]


# ----------------------------------------------------------------- metrics
def instances_from_prob(prob, thr=0.5, min_size=50):
    lab, n = ndi.label(prob > thr)
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        for i, s in enumerate(sizes, 1):
            if s < min_size:
                lab[lab == i] = 0
    return lab


def seg_score(pred_lab, gt_lab):
    """Cell-Tracking-Challenge SEG: mean Jaccard of GT objects matched with |R∩S|/|R|>0.5."""
    ids = np.unique(gt_lab)
    ids = ids[ids > 0]
    scores = []
    for g in ids:
        gm = gt_lab == g
        ov = pred_lab[gm]
        ov = ov[ov > 0]
        if ov.size == 0:
            scores.append(0.0)
            continue
        pm = pred_lab == np.bincount(ov).argmax()
        inter = (gm & pm).sum()
        scores.append(inter / (gm | pm).sum() if inter / gm.sum() > 0.5 else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def fg_iou(prob, gt_lab, thr=0.5):
    p, g = prob > thr, gt_lab > 0
    return float((p & g).sum() / max((p | g).sum(), 1))
