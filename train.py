"""Train / evaluate U-Net on a Cell Tracking Challenge 2D dataset (paper's Table 2).

Example (Colab T4):
    python train.py --dataset PhC-C2DH-U373 --iters 6000
    python train.py --dataset DIC-C2DH-HeLa --iters 6000
Ablations:  --no_elastic   --no_weight
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import (TileSampler, download_ctc, fg_iou, instances_from_prob, load_ctc,
                  predict_full, seg_score)
from unet import UNet, count_conv_layers, valid_output_size


def evaluate(model, items, device, amp, tta):
    segs, ious, probs = [], [], []
    for it in items:
        prob = predict_full(model, it["img"], device, amp=amp, tta=tta)
        segs.append(seg_score(instances_from_prob(prob), it["lab"]))
        ious.append(fg_iou(prob, it["lab"]))
        probs.append(prob)
    return float(np.nanmean(segs)), float(np.mean(ious)), probs


def save_figure(items, probs, path, n=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(n, len(items))
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for r in range(n):
        ax[r, 0].imshow(items[r]["img"], cmap="gray"); ax[r, 0].set_title("input")
        ax[r, 1].imshow(items[r]["img"], cmap="gray")
        ax[r, 1].contour(items[r]["lab"] > 0, [0.5], colors="yellow", linewidths=1); ax[r, 1].set_title("ground truth")
        ax[r, 2].imshow(items[r]["img"], cmap="gray")
        ax[r, 2].imshow(np.ma.masked_less(probs[r], 0.5), alpha=0.5, cmap="cool")
        ax[r, 2].contour(items[r]["lab"] > 0, [0.5], colors="yellow", linewidths=0.8); ax[r, 2].set_title("u-net (cyan) vs GT (yellow)")
        for a in ax[r]: a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PhC-C2DH-U373")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.99)
    ap.add_argument("--base", type=int, default=64, help="channels of first block (paper: 64)")
    ap.add_argument("--split", choices=["seq", "random"], default="seq",
                    help="seq: train on sequence 01, test on 02; random: 80/20 over frames")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--no_elastic", action="store_true")
    ap.add_argument("--no_weight", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (dev.type == "cuda") and not a.no_amp
    os.makedirs(a.out, exist_ok=True)
    name = a.dataset + (f"_{a.tag}" if a.tag else "")

    path = download_ctc(a.dataset, a.data_root)
    items = load_ctc(path)
    train = [i for i in items if i["name"].startswith("01")]
    test = [i for i in items if i["name"].startswith("02")]
    if a.split == "random" or not train or not test:
        rng = random.Random(a.seed)
        idx = list(range(len(items))); rng.shuffle(idx)
        k = max(1, int(0.2 * len(items)))
        test = [items[i] for i in idx[:k]]; train = [items[i] for i in idx[k:]]
        a.split = "random"
    print(f"annotated frames: {len(items)} | train {len(train)} | test {len(test)} | split={a.split}")

    model = UNet(1, 2, base=a.base).to(dev)
    print(f"conv layers: {count_conv_layers(model)} (paper: 23) | 572->{valid_output_size(572)} | "
          f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M | device {dev} amp={amp}")

    sampler = TileSampler(train)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.7 * a.iters)], 0.1)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)

    log, t0, run = [], time.time(), 0.0
    for it in range(1, a.iters + 1):
        model.train()
        x, y, w = sampler.sample(dev, elastic=not a.no_elastic, use_weight=not a.no_weight)
        with torch.autocast(device_type=dev.type, enabled=amp):
            logits = model(x)
        loss = (F.cross_entropy(logits.float(), y, reduction="none") * w).mean()   # weighted CE, Eq. (1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % 100 == 0:
            log.append((it, run))
        if it % a.eval_every == 0 or it == a.iters:
            seg, iou, _ = evaluate(model, test, dev, amp, tta=False)
            print(f"it {it:5d} | loss {run:.4f} | test SEG {seg:.4f} | fg-IoU {iou:.4f} | {time.time()-t0:.0f}s")

    seg, iou, probs = evaluate(model, test, dev, amp, tta=True)
    print(f"FINAL (8-fold flip/rot TTA) test SEG(IOU) {seg:.4f} | fg-IoU {iou:.4f}")
    res = dict(dataset=a.dataset, tag=a.tag, seg=seg, fg_iou=iou, iters=a.iters, base=a.base,
               split=a.split, n_train=len(train), n_test=len(test), elastic=not a.no_elastic,
               weight_map=not a.no_weight, train_seconds=time.time() - t0, loss_log=log)
    json.dump(res, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
    torch.save(model.state_dict(), os.path.join(a.out, name + ".pt"))
    save_figure(test, probs, os.path.join(a.out, name + "_qualitative.png"))


if __name__ == "__main__":
    main()
