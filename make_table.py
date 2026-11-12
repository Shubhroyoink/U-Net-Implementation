"""Collect results/*.json into a markdown table next to the paper's Table 2 numbers."""
import glob
import json

PAPER = {"PhC-C2DH-U373": 0.9203, "DIC-C2DH-HeLa": 0.7756}
PAPER_2ND = {"PhC-C2DH-U373": 0.83, "DIC-C2DH-HeLa": 0.46}

rows = [json.load(open(f)) for f in sorted(glob.glob("results/*.json"))]
print("| Dataset | Variant | Iters | Base ch. | Split (train/test) | Ours SEG(IOU) | Ours fg-IoU | Paper u-net | Paper 2nd best |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    variant = r["tag"] or "full (weight map + elastic)"
    print(f"| {r['dataset']} | {variant} | {r['iters']} | {r['base']} | {r['split']} ({r['n_train']}/{r['n_test']}) | "
          f"{r['seg']:.4f} | {r['fg_iou']:.4f} | {PAPER[r['dataset']]} | {PAPER_2ND[r['dataset']]} |")
