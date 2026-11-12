# U-Net (Ronneberger et al., 2015) — PyTorch re-implementation

Re-implementation of *U-Net: Convolutional Networks for Biomedical Image Segmentation* (arXiv:1505.04597),
written from the paper. Target of reproduction: **Table 2** (ISBI Cell Tracking Challenge 2015, PhC-U373 and DIC-HeLa),
scaled for a free Colab **T4** GPU.

## Files
| File | What it is |
|---|---|
| `unet.py` | The U-Net: unpadded 3x3 convs, copy-and-crop skips, 2x2 up-conv, dropout, He init. 572→388, 23 conv layers (asserted) |
| `data.py` | Core training ideas: 1-px separation borders, weight map (Eq. 2), elastic deformation (3x3 grid, σ=10 px, bicubic), overlap-tile inference with mirroring (Fig. 2), SEG/IoU metric |
| `train.py` | SGD, momentum 0.99, batch 1, weighted pixel-wise softmax cross-entropy (Eq. 1); evaluation, figure, JSON results |
| `make_table.py` | Builds the "ours vs paper" markdown table from `results/*.json` |
| `unet_colab.ipynb` | One-click Colab notebook running everything |
| `REPORT.md` | Report template — fill in the numbers after running |

## Run on Colab (T4)
1. Runtime → Change runtime type → **T4 GPU**.
2. Upload `unet_colab.ipynb` and run all cells. (Datasets are downloaded from celltrackingchallenge.net; if the
   download is blocked, download `PhC-C2DH-U373.zip` / `DIC-C2DH-HeLa.zip` manually, unzip into `data/`.)
3. `python make_table.py` prints the comparison table for the report.

Rough budget on a T4 (fp16 autocast): ~6000 iterations per run, on the order of 10–20 min — check the printed timing.
If you hit out-of-memory or time limits, use `--base 32` (half width) or fewer `--iters`.

## What is faithful, and what is changed (state this in the report)
Faithful: architecture (Fig. 1), unpadded convs + crop, overlap-tile with mirroring, weight map with w0=10, σ=5,
elastic deformation (3x3 grid, σ=10), He init, batch size 1 + momentum 0.99, dropout at end of contracting path.

Changed / not available:
- **Test set**: the challenge's test labels are secret and the paper's numbers come from the organisers' server.
  We evaluate on **held-out annotated gold-truth frames** (train on sequence 01, test on sequence 02), so the numbers are
  *not* directly comparable to the paper's leaderboard values. Report this clearly.
- **Metric**: we re-implement the challenge's SEG (mean Jaccard of matched objects, match if |R∩S|/|R|>0.5) and also report foreground IoU.
  Instances come from connected components of the predicted foreground (separation borders split touching cells).
- **Training length**: ~6000 iterations vs the paper's ~10 h on a Titan GPU. LR/schedule are our choice (paper gives none):
  lr 1e-3, ×0.1 at 70%, gradient clipping 10, mixed precision.
- **Loss normalisation**: weighted CE is averaged over pixels (the paper writes a sum).
- **Up-convolution** is a 2x2 stride-2 transposed convolution.
- Final numbers use 8-fold flip/rot90 test-time augmentation.
- Optional ablations (`--no_elastic`, `--no_weight`) are extra, not in the paper's Table 2.

## References / sources
- Paper: O. Ronneberger, P. Fischer, T. Brox, *U-Net*, MICCAI 2015, arXiv:1505.04597.
- Data: Cell Tracking Challenge (Maška et al., Bioinformatics 2014), http://celltrackingchallenge.net — PhC-C2DH-U373, DIC-C2DH-HeLa.
- **No official or third-party U-Net repository was consulted**; the code was written from the paper.
  If you consult any (e.g. the authors' Caffe release), cite it here and in the report.
- Building blocks used: PyTorch (`nn.Conv2d`, `nn.ConvTranspose2d`, SGD), SciPy (distance transform, labeling), tifffile.
