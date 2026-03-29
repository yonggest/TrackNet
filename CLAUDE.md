# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unofficial PyTorch implementation of [TrackNet](https://arxiv.org/abs/1907.03698) — a U-Net-style encoder-decoder that detects tennis balls in broadcast video by processing 3 consecutive frames and predicting Gaussian heatmaps.

## Common Commands

### Data Preparation
```bash
# Generate ground truth heatmaps and train/val CSV labels from raw dataset
python gt_gen.py --path_input ./Dataset --path_output ./datasets/trackNet
```

### Training
```bash
python main.py [--batch_size 2] [--exp_id default] [--num_epochs 500] [--lr 1.0] [--val_intervals 5] [--steps_per_epoch 200]
```
Checkpoints saved to `./exps/{exp_id}/model_last.pt` and `model_best.pt` (best F1). TensorBoard logs in `./exps/{exp_id}/plots/`.

### Evaluation
```bash
python test.py --batch_size 2 --model_path ./exps/default/model_best.pt
```

### Video Inference
```bash
python infer_on_video.py --model_path ./exps/default/model_best.pt --video_path input.mp4 --video_out_path output.avi [--extrapolation]
```

## Architecture

### Model (`model.py`)
`BallTrackerNet` is a U-Net encoder-decoder:
- **Input:** 9 channels — 3 consecutive RGB frames concatenated along the channel axis
- **Encoder:** 3 max-pool stages (64→128→256→512 feature channels)
- **Decoder:** 3 upsample stages with skip connections (512→256→128→64)
- **Output:** 256-channel heatmap reshaped to `(batch, 256, H×W)`, treated as pixel-wise multi-class logits
- Training: raw logits for `CrossEntropyLoss`; inference: softmax applied

### Data Pipeline
1. **`gt_gen.py`** preprocesses the raw `Dataset/` into `datasets/trackNet/`:
   - `images/`: downsampled frames (360×640)
   - `gts/`: Gaussian heatmaps (kernel size=20, variance=10) as supervision targets
   - `labels_train.csv` / `labels_val.csv`: 70/30 split with paths + coordinates + visibility

2. **`datasets.py`** (`trackNetDataset`): loads 3 consecutive frames + corresponding heatmap + (x, y, visibility). Frames normalized to [0, 1].

3. **`general.py`**: `train()` / `validate()` loops; `postprocess()` converts heatmap output → ball coordinates via threshold (127) + Hough Circle detection (output is scaled ×2 back to 1280×720).

### Inference Pipeline (`infer_on_video.py`)
Video frames → sliding 3-frame window → model → Hough Circle postprocess → outlier removal (>100px jump) → optional linear interpolation for gaps ≤4 frames → render with 7-frame trail.

### Optimizer
Adadelta (default lr=1.0). Validation metric is F1 score (TP threshold: ≤5px Euclidean distance). Metrics are tracked per visibility class (1, 2, 3).

## Dataset

Raw dataset lives in `Dataset/game{1-10}/Clip{1-13}/` — numbered JPG frames + `Label.csv` (columns: `file name`, `visibility`, `x-coordinate`, `y-coordinate`, `status`). The full dataset (19,835 labeled frames, 1280×720 @ 30fps) is available on Google Drive (see README.md).
