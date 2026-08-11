# GeoSeg-SAM

GeoSeg-SAM is a SAM-based framework for closed-set semantic segmentation of geological environments in remote sensing imagery. The repository provides the training pipeline, model implementation, and configurations for the WLK and YJS datasets.

Repository: `https://github.com/WanZhan-lucky/GeoSeg-SAM`

## Overview

GeoSeg-SAM adapts the SAM ViT-B architecture to multi-class geological semantic segmentation. The default workflow reads the experiment settings from a YAML file, builds the dataset and model, loads the SAM pretrained checkpoint, trains the selected parameters, evaluates the model on the validation set, and saves checkpoints and evaluation results.

The main implementation is organized around the following files:

```text
train.py
└── models/sam.py
    └── models/mmseg/models/sam/
        ├── __init__.py
        ├── image_encoder.py
        ├── mask_decoder.py
        └── transformer.py
```

## Repository Structure

```text
GeoSeg-SAM/
├── configs/
│   ├── wlk-256input.yaml
│   └── yjs-256input.yaml
├── datasets/
│   ├── image_folderw.py
│   ├── wrappersw.py
│   └── wrappersY.py
├── models/
│   ├── sam.py
│   └── mmseg/models/sam/
│       ├── __init__.py
│       ├── image_encoder.py
│       ├── mask_decoder.py
│       └── transformer.py
├── pretrained/
│   └── sam_vit_b_01ec64.pth
├── wdatas/
│   ├── wlk-256data/
│   └── yjs-256data/
├── requirements.txt
└── train.py
```

The `pretrained/`, `wdatas/`, and output directories are prepared locally and are not required to be stored in the repository.

## Main Workflow

### 1. Training entry: `train.py`

`train.py` is the main executable file. It performs the following steps:

1. Loads the selected YAML configuration.
2. Builds paired image-label datasets through `datasets/image_folderw.py` and applies the configured train/validation wrapper.
3. Creates the GeoSeg-SAM model through the model registry.
4. Loads the SAM ViT-B pretrained checkpoint.
5. Freezes most image-encoder parameters and enables the task-specific trainable modules.
6. Trains the model and evaluates it on the validation set.
7. Records segmentation metrics and saves checkpoints and confusion-matrix heatmaps.

### 2. Model assembly: `models/sam.py`

`models/sam.py` defines and registers the GeoSeg-SAM model. It connects the image encoder, mask decoder, and two-way transformer into the semantic-segmentation network.

The model uses empty sparse prompts and a learned no-mask embedding, allowing the SAM architecture to operate as a closed-set semantic segmentation model. The active training objective combines Cross-Entropy loss and Dice loss.

### 3. SAM module exports: `models/mmseg/models/sam/__init__.py`

This file exposes the SAM components used by `models/sam.py`, including:

- `ImageEncoderViT`
- `MaskDecoder`
- `TwoWayTransformer`

### 4. Image encoder: `image_encoder_DSSA.py`

The image encoder is based on the SAM ViT image encoder. It includes adapter-related and multi-scale feature-processing modules for geological remote-sensing imagery.

The encoder converts an input image into the main image embedding and intermediate multi-level features used by the segmentation model.

### 5. Mask decoder: `mask_decoder.py`

The mask decoder transforms the encoded image representation into class-wise segmentation masks. It combines image features, positional encoding, prompt embeddings, and transformer outputs to produce the final semantic prediction.

### 6. Two-way transformer: `transformer.py`

The two-way transformer performs attention-based interaction between image features and token embeddings. Its output is used by the mask decoder to generate the segmentation masks and mask-quality predictions.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/WanZhan-lucky/GeoSeg-SAM.git
cd GeoSeg-SAM
```

Create the environment:

```bash
conda create -n geoseg-sam python=3.8 -y
conda activate geoseg-sam
```

Install PyTorch and TorchVision according to the local CUDA environment. For example:

```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install rasterio seaborn prettytable
```

The bundled MMSegmentation implementation uses MMCV 1.x. Install an MMCV build compatible with the selected PyTorch and CUDA versions.

## Pretrained Checkpoint

Prepare the official SAM ViT-B checkpoint and place it at:

```text
pretrained/sam_vit_b_01ec64.pth
```

The provided configurations use:

```yaml
sam_checkpoint: ./pretrained/sam_vit_b_01ec64.pth
```

## Dataset Preparation

The repository provides configurations for the WLK and YJS datasets. Dataset download links are not included in this README. Prepare the datasets locally using the following structure:

```text
wdatas/
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

Data conventions:

- Images are three-band raster images and are converted to RGB by the data loader.
- Labels are single-band class-index masks.
- WLK contains 13 classes with label indices from `0` to `12`.
- YJS contains 8 classes with label indices from `0` to `7`.
- Image and label directories must contain correctly paired files.
- The provided configurations resize images and labels to `256 × 256`.

Dataset paths and class definitions can be changed in the corresponding YAML configuration file.

## Training

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

### Optional run tag

```bash
python train.py \
  --config configs/wlk-256input.yaml \
  --path ./outputs \
  --name geoseg_wlk \
  --tag run1
```

This creates the experiment directory:

```text
outputs/geoseg_wlk_run1/
```

### Command-line arguments

| Argument | Description |
|---|---|
| `--config` | Path to the YAML configuration file. |
| `--path` | Root directory used to save experiment outputs. |
| `--name` | Experiment directory name. |
| `--tag` | Optional suffix appended to the experiment name. |
| `--local_rank` | Compatibility argument retained by the training script. |

It is recommended to explicitly specify `--path ./outputs` when starting an experiment.

## Configuration

The main experiment settings are stored in `configs/wlk-256input.yaml` and `configs/yjs-256input.yaml`.

Important fields include:

```yaml
train_dataset:
  dataset:
    name: paired-image-folders  # registered in datasets/image_folderw.py
    args:
      classes: [...]
      root_path_1: ./wdatas/.../train/images
      root_path_2: ./wdatas/.../train/labels
  wrapper:
    name: train                 # train wrapper
    args:
      inp_size: 256
      augment: true
  batch_size: 1

val_dataset:
  dataset:
    name: paired-image-folders  # registered in datasets/image_folderw.py
    args:
      classes: [...]
      root_path_1: ./wdatas/.../val/images
      root_path_2: ./wdatas/.../val/labels
  wrapper:
    name: val                   # validation wrapper
    args:
      inp_size: 256
  batch_size: 1

sam_checkpoint: ./pretrained/sam_vit_b_01ec64.pth

model:
  name: sam
  args:
    num_classes: 13
    inp_size: 256

optimizer:
  name: adamw
  args:
    lr: 0.0002

epoch_max: 120
```

The dataset pipeline is divided into two parts:

- `datasets/image_folderw.py` registers `image-folder` and `paired-image-folders`, reads the raster images and masks, and pairs the image and label folders.
- `datasets/wrappersw.py` and `datasets/wrappersY.py` provide the `train` and `val` preprocessing wrappers. The wrapper enabled in `datasets/__init__.py` is selected through `wrapper.name` in the YAML configuration.

The default dataset initialization currently imports `image_folderw.py` and `wrappersw.py`. `wrappersY.py` is retained as the alternative wrapper implementation for the corresponding data-processing setup.

For YJS, `num_classes` is set to `8`. When adapting the project to another dataset, keep the class list, label indices, and `num_classes` consistent.

## Outputs

For an experiment named `geoseg_wlk`, the main outputs are:

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

The validation process reports overall accuracy, mIoU, per-class IoU, precision, recall, F1 score, FWIoU, and the confusion matrix. The best checkpoint is selected according to validation mIoU.

## Citation

If you use GeoSeg-SAM or the associated datasets in your research, please cite the corresponding paper. Citation information will be added after publication.

## Acknowledgements

This project builds upon the Segment Anything Model, MMSegmentation, and related open-source research. We thank the respective authors and contributors for making their work available.
