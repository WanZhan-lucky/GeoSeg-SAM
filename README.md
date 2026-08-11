# GeoSeg-SAM

GeoSeg-SAM is a SAM-based adaptation framework for closed-set semantic segmentation of geological environments in remote sensing imagery. This repository contains the training code and configurations for the WLK and YJS datasets.

## Repository Structure

```text
GeoSeg-SAM/
├── configs/
│   ├── wlk-256input.yaml
│   └── yjs-256input.yaml
├── datasets/
├── models/
├── requirements.txt
├── train.py
├── pretrained/                  # create manually
│   └── sam_vit_b_01ec64.pth
├── wdatas/                      # create manually
│   ├── wlk-256data/
│   └── yjs-256data/
└── outputs/                     # generated during training
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/WanZhan-lucky/GeoSeg_SAM.git
cd GeoSeg_SAM
```

A CUDA-enabled environment is required by the current training script. One compatible setup is:

```bash
conda create -n geoseg-sam python=3.8 -y
conda activate geoseg-sam

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install rasterio seaborn prettytable
```

The bundled MMSegmentation code requires MMCV 1.x (`1.1.4 <= mmcv <= 1.7.0`). Install an MMCV or MMCV-Full build compatible with the selected PyTorch and CUDA versions. Do not install MMCV 2.x for the current codebase.

## SAM Pretrained Checkpoint

Download the official SAM ViT-B checkpoint:

- [`sam_vit_b_01ec64.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

Place it at:

```text
GeoSeg-SAM/
└── pretrained/
    └── sam_vit_b_01ec64.pth
```

Both provided configurations use:

```yaml
sam_checkpoint: ./pretrained/sam_vit_b_01ec64.pth
```

## Dataset Preparation

Arrange the datasets as follows:

```text
GeoSeg-SAM/
└── wdatas/
    ├── wlk-256data/
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   └── val/
    │       ├── images/
    │       └── labels/
    └── yjs-256data/
        ├── train/
        │   ├── images/
        │   └── labels/
        └── val/
            ├── images/
            └── labels/
```

The active data loader uses `rasterio`:

- Images are read as three-band raster images and converted to RGB.
- Labels are read from the first raster band as single-channel class-index masks.
- WLK labels must use class indices `0-12` for 13 classes.
- YJS labels must use class indices `0-7` for 8 classes.
- Image and label files are paired according to their sorted file order, so the two directories must contain matching files in the same naming order.
- Images and masks are resized to `256 × 256` by the provided configurations.

## Training

> [!IMPORTANT]
> The current repository snapshot cannot run training directly: `train.py` imports missing root-level `utils.py` and `eval_iou.py`, and the bundled SAM package also references a missing `models/mmseg/models/sam/prompt_encoder.py`. These files/imports must be restored or corrected before training can start.

### WLK

```bash
python train.py \
  --config configs/wlk-256input.yaml \
  --path ./outputs \
  --name geoseg_wlk
```

### YJS

```bash
python train.py \
  --config configs/yjs-256input.yaml \
  --path ./outputs \
  --name geoseg_yjs
```

### Command-Line Arguments

| Argument | Description |
|---|---|
| `--config` | Path to the YAML configuration file. Defaults to `configs/wlk-256input.yaml`. |
| `--path` | Root directory for training outputs. Always set this explicitly because the code's default path is machine-specific. |
| `--name` | Experiment directory name. If omitted, the configuration filename is used. |
| `--tag` | Optional suffix appended to the experiment name. |
| `--local_rank` | Compatibility argument retained by the script; it is not used by the current single-process training code. |

Example with a run tag:

```bash
python train.py \
  --config configs/wlk-256input.yaml \
  --path ./outputs \
  --name geoseg_wlk \
  --tag run1
```

This creates `./outputs/geoseg_wlk_run1/`.

## Outputs

For an experiment named `geoseg_wlk`, the training script writes:

```text
outputs/
├── geoseg_wlk/
│   ├── config.yaml
│   ├── model_epoch_last.pth
│   ├── optim_epoch_last.pth
│   ├── model_epoch_best.pth
│   └── model_epoch_best_mIoU_*.pth
└── heatmap/
    └── heatmap_<epoch>.jpg
```

- `model_epoch_last.pth` and `optim_epoch_last.pth` are updated after each epoch.
- `model_epoch_best.pth` is updated when validation mIoU improves.
- `model_epoch_best_mIoU_<score>.pth` is additionally saved when the best mIoU is greater than `0.61`.
- A confusion-matrix heatmap is saved under `<--path>/heatmap/` when a new best mIoU is reached.

Validation reports overall accuracy, mIoU, per-class IoU, precision, recall, F1 score, FWIoU, and the confusion matrix.

## Configuration

The main settings are defined in `configs/wlk-256input.yaml` and `configs/yjs-256input.yaml`:

```yaml
train_dataset:
  dataset:
    args:
      classes: [...]
      root_path_1: ./wdatas/.../train/images
      root_path_2: ./wdatas/.../train/labels
  wrapper:
    args:
      inp_size: 256
  batch_size: 1

val_dataset:
  dataset:
    args:
      classes: [...]
      root_path_1: ./wdatas/.../val/images
      root_path_2: ./wdatas/.../val/labels
  batch_size: 1

sam_checkpoint: ./pretrained/sam_vit_b_01ec64.pth

model:
  name: sam
  args:
    num_classes: 13  # 8 for YJS
    inp_size: 256

optimizer:
  name: adamw
  args:
    lr: 0.0002

epoch_max: 120
```

The current training implementation freezes most image-encoder parameters and trains the adapter-related modules and decoder. Although the configuration uses `loss: iou`, the active loss calculation in `models/sam.py` is Cross-Entropy plus Dice loss.

To adapt the project to another dataset, update the image and label paths, class names, `num_classes`, and other task-specific settings consistently.

## Notes

- The default data paths are relative to the repository root under `./wdatas/`.
- The default SAM checkpoint path is `./pretrained/sam_vit_b_01ec64.pth`.
- Explicitly pass `--path ./outputs`; the default value in `train.py` is specific to the original development environment.
- The current code uses `.cuda()` directly in several places, so training requires a CUDA-capable GPU.
- The repository currently provides training and validation code but no standalone inference script.

## Citation

If you use this code or the released datasets in your research, please cite the corresponding GeoSeg-SAM paper. The BibTeX entry will be added after publication.

## Acknowledgements

This work builds upon the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), MMSegmentation, and related open-source research. We thank their authors and contributors for making these resources publicly available.
