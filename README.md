# Kura Clover — Semi-Supervised Segmentation

Semi-supervised semantic segmentation of Kura clover in PVC quadrats using FlexMatch with a DeepLabV3+ backbone.

**Classes (6):** `soil` · `quadrat` · `clover_leaves` · `clover_stems` · `person` · `other_veg`

**Data split:** 9 labeled training images · 1 labeled validation image · 1125 unlabeled images

---

## How it works

Training uses [FlexMatch](https://arxiv.org/abs/2110.08263) — a semi-supervised algorithm that generates pseudo-labels for unlabeled images using class-adaptive confidence thresholds:

1. **Labeled pass** — standard cross-entropy / TvmfDice loss on the 9 labeled images
2. **Unlabeled weak pass** — model runs in eval mode on lightly-augmented unlabeled images to produce clean pseudo-label probabilities
3. **Unlabeled strong pass** — same image with heavy augmentation (color jitter, glass blur, channel shuffle) is run in train mode; pseudo-labels above the per-class threshold supervise this pass
4. **EMA** — an Exponential Moving Average of model weights is maintained and used during validation

Inference on full-resolution images (3456×5184) uses a 1024×1024 sliding window with 50% overlap; softmax probabilities are averaged across overlapping tiles.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Data layout expected:**
```
data/
  labeled/
    images/   # 9 training images
    targets/  # corresponding grayscale class-index masks (0–5)
  val/
    images/   # 1 validation image
    targets/
  unlabeled/
    images/   # 1125 unlabeled images
metadata/
  segmentation_class_map.json
  dataset_norm.json
```

---

## Training

### Local smoke test (CPU, Mac)

```bash
python train_semisup.py --config configs/train_semisup_config_local.yaml
```

### Server (2× RTX 5090, NCCL)

```bash
torchrun --standalone --nproc-per-node=2 train_semisup.py --config configs/train_semisup_config.yaml --backend nccl
```

The default config is at [`configs/train_semisup_config.yaml`](configs/train_semisup_config.yaml).

---

## Checkpoints

The top-5 checkpoints by validation loss are saved automatically to the directory set in `conf.directories.checkpoint_dir`. Each checkpoint contains the model state dict and, if EMA is enabled, the EMA shadow parameters.

---

## Inference

```bash
python infer.py --config configs/train_semisup_config.yaml --checkpoint <path/to/checkpoint.pt> --image <path/to/image.jpg> --output <output_mask.png>
```

Produces a grayscale PNG where each pixel value is the predicted class index (0–5).

---

## Model

- **Architecture:** DeepLabV3+ with EfficientNet-B1 encoder (`segmentation_models_pytorch`)
- **Loss:** TvmfDice loss (t-vMF kernel, adaptive κ) combined with cross-entropy
- **Optimizer:** AdamW with cosine annealing LR schedule and linear warmup
- **EMA decay:** 0.9

---

## Class map

| Index | Class |
|-------|-------|
| 0 | soil |
| 1 | quadrat |
| 2 | clover_leaves |
| 3 | clover_stems |
| 4 | person |
| 5 | other_veg |
