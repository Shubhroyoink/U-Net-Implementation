# U-Net in PyTorch: reproducing Table 2 (Cell Tracking Challenge)

A from-scratch PyTorch re-implementation of **U-Net** (Ronneberger, Fischer, Brox, MICCAI 2015, [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)),
evaluated on the 2D Cell Tracking Challenge datasets used in **Table 2** of the paper (PhC-C2DH-U373 and DIC-C2DH-HeLa).
Everything is scaled to fit a free Colab **T4** GPU, so results are a reduced-scale reproduction, not an exact one.

## Repository structure

| File | Purpose |
|---|---|
| `unet.py` | Model: unpadded 3x3 convs + ReLU, 2x2 max-pool, 2x2 stride-2 transposed conv, copy-and-crop skip connections, dropout, He initialisation. 572x572 input gives 388x388 output; 23 conv layers (19 conv + 4 up-conv). |
| `data.py` | Data pipeline: dataset download, 1-px separation borders between touching cells, weight map (Eq. 2), GPU elastic deformation, overlap-tile inference with mirror padding, SEG / IoU metrics. |
| `train.py` | Training and evaluation: SGD, mixed precision, LR schedule, 8-fold test-time augmentation, JSON/checkpoint/figure output. |
| `make_table.py` | Collects `results/*.json` into a markdown table next to the paper's numbers. |
| `unet_colab.ipynb` | Notebook that writes these files and runs all experiments on Colab. |

## What is implemented from the paper

1. **Architecture (Fig. 1):** contracting path (64-128-256-512, bottom 1024), symmetric expanding path, centre-cropped skip connections, no fully connected layers, 1x1 output conv. Dropout at the end of the contracting path.
2. **Weight map (Eq. 2):** `w(x) = w_c(x) + w0 * exp(-(d1(x)+d2(x))^2 / (2 sigma^2))` with `w0 = 10`, `sigma = 5`. `w_c` balances class frequencies; `d1`, `d2` are distances to the nearest and second-nearest cell. Touching cells are separated by a 1-px background border so the loss can emphasise the gap.
3. **Elastic deformation (Sec. 3.1):** random displacements on a coarse 3x3 grid (sigma = 10 px), bicubic interpolation; the same deformation is applied to image, mask and weight map. Also random flips, 90-degree rotations and gray-value jitter.
4. **Overlap-tile strategy (Fig. 2):** input is mirror-padded, 572x572 tiles are predicted with stride 388, and the 388x388 outputs are stitched. This allows arbitrarily large images despite limited GPU memory.
5. **Optimisation:** weighted pixel-wise softmax cross-entropy (Eq. 1), batch size 1, momentum 0.99.

## Deviations from the paper (read before comparing numbers)

| Item | Paper | This work |
|---|---|---|
| Test data | Secret test set, scored by the challenge organisers | Held-out annotated gold-truth frames: train on sequence 01, test on sequence 02 |
| Training length | About 10 h on a Titan GPU | 6000 iterations on a T4 |
| LR schedule | Not specified | lr 1e-3, x0.1 at 70% of training, gradient clipping at 10 |
| Precision | fp32 | fp16 autocast (mixed precision) |
| Loss normalisation | Sum over pixels | Mean over pixels |
| Up-convolution | Upsample + 2x2 conv | 2x2 stride-2 transposed conv |
| Instance extraction | Not described | Threshold 0.5 + connected components (objects under 50 px removed) |
| Test-time augmentation | Rotated copies (EM task) | 8-fold flip/rot90 averaging |
| Seeds | Not reported | Single seed (0) |

**Metrics.** *SEG* is the challenge's Jaccard measure (each ground-truth cell is matched to a predicted object covering more than 50% of it; the score is their IoU, or 0 if unmatched). *fg-IoU* is plain foreground/background IoU. I read the paper's "IOU" as SEG.

## Results

| Dataset | Variant | Iters | Base ch. | Train/Test frames | **SEG (ours)** | fg-IoU (ours) | Paper U-Net | Paper 2nd best |
|---|---|---|---|---|---|---|---|---|
| PhC-C2DH-U373 | Full (weight map + elastic) | 6000 | 64 | 15 / 19 | **0.7720** | 0.7361 | 0.9203 | 0.83 |
| PhC-C2DH-U373 | No elastic deformation | 6000 | 64 | 15 / 19 | 0.7637 | 0.7320 | n/a | n/a |
| PhC-C2DH-U373 | No weight map | 6000 | 64 | 15 / 19 | 0.8107 | 0.7672 | n/a | n/a |
| DIC-C2DH-HeLa | Full (weight map + elastic) | 6000 | 64 | 9 / 9 | **0.1635** | 0.8261 | 0.7756 | 0.46 |

Paper numbers come from the challenge leaderboard on a hidden test set, so they are **not directly comparable** with ours.

### Discussion

- **PhC-U373:** SEG of 0.772 is below the paper's 0.9203 and also below the paper's second-best entry (0.83). With 4 to 5 times fewer iterations, a different test set and untuned hyperparameters, a gap is expected, but this run does not match the paper.
- **DIC-HeLa:** fg-IoU is high (0.826) while SEG is very low (0.1635). This suggests the foreground is segmented well but individual cells are not separated. A likely cause is that dense, touching HeLa cells merge into one component in the connected-components step. I have not verified this; checking the predicted instance count against ground truth, or replacing the instance step with a watershed on the probability map, would test it. It is a limitation of my post-processing, not necessarily of the network.
- **Ablations:** elastic deformation helped slightly (0.7720 vs 0.7637), as the paper argues. Removing the weight map did **not** hurt; it scored higher (0.8107). With one seed and a small test set, differences of this size may be noise, so these ablations do not confirm or refute the paper's claim about the weight map. Repeating with several seeds would be needed.

## Getting started

Requirements: Python 3.9+, PyTorch, SciPy, NumPy, Matplotlib, `tifffile` and `imagecodecs` (the dataset TIFFs are LZW-compressed).

```bash
pip install tifffile imagecodecs scipy matplotlib
```

Quick check (about a minute on a T4):

```bash
python train.py --dataset PhC-C2DH-U373 --iters 200 --eval_every 100 --tag test
```

Full runs and ablations:

```bash
python train.py --dataset PhC-C2DH-U373 --iters 6000
python train.py --dataset DIC-C2DH-HeLa --iters 6000
python train.py --dataset PhC-C2DH-U373 --iters 6000 --no_elastic --tag no_elastic
python train.py --dataset PhC-C2DH-U373 --iters 6000 --no_weight  --tag no_weight
python make_table.py
```

Datasets are downloaded automatically from celltrackingchallenge.net into `data/`. If the download is blocked, download the zips manually and unzip them into `data/`. Outputs (JSON metrics, checkpoints, qualitative figures) go to `results/`. If memory or time is tight, use `--base 32` or fewer `--iters`.

## References

- O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015. arXiv:1505.04597.
- M. Maska et al. *A benchmark for comparison of cell tracking algorithms.* Bioinformatics 30, 2014. Data: http://celltrackingchallenge.net
- Libraries: PyTorch, SciPy, tifffile.

No official or third-party U-Net repository was consulted; the code was written from the paper.
