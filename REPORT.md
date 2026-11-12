# Reproducing U-Net (Table 2) on a Colab T4

## 1. Paper and target
U-Net (Ronneberger, Fischer, Brox, 2015): contracting/expanding path with skip connections, trained with heavy elastic
augmentation and a border-weighted loss. We reproduce **Table 2**: segmentation IOU on the ISBI Cell Tracking Challenge
2D datasets PhC-U373 and DIC-HeLa.

## 2. Implementation summary
(Describe unet.py / data.py: architecture, weight map Eq. 2, elastic deformation, overlap-tile, SEG metric. See README.)
Sanity checks done: 572→388 output, 23 conv layers, weight ≈ w0 in 1-px gap between touching cells.

## 3. What was changed vs the paper
| Item | Paper | Ours | Reason |
|---|---|---|---|
| Test data | secret test set, organiser server | held-out gold frames (seq 01 train / seq 02 test) | labels not public |
| Training | ~10 h, Titan GPU | ~6000 iters on T4 (fill in actual minutes) | compute |
| Precision | fp32 | fp16 autocast | speed |
| LR schedule | not given | 1e-3, ×0.1 at 70% | our choice |
| Eval | – | 8-fold flip/rot TTA | cheap gain |
| Model width | 64 base channels | (fill in: 64 or 32) | memory/time |

## 4. Results
Paste the output of `python make_table.py` here.

| Dataset | Ours SEG(IOU) | Paper u-net | Paper 2nd best |
|---|---|---|---|
| PhC-U373 | _fill_ | 0.9203 | 0.83 |
| DIC-HeLa | _fill_ | 0.7756 | 0.46 |

Include `results/*_qualitative.png` (cf. paper Fig. 4) and the loss curves from `loss_log` in the JSON files.

## 5. Discussion
- How close are we to the paper? Possible causes of gaps: different (held-out, not challenge) test set, far fewer iterations,
  our instance extraction from foreground components, different LR schedule.
- Optional ablation (`--no_weight`, `--no_elastic`): effect of the weight map and elastic augmentation.
- Limitations: single seed, tiny test set (state its size).

## 6. Sources
Paper arXiv:1505.04597; Cell Tracking Challenge data; PyTorch/SciPy/tifffile. No external U-Net repository consulted (update if that changes).
