# COMP9517 iNaturalist species classification

This repository contains our code for the COMP9517 group project. The main comparison is between a ResNet-18 trained from random initialisation and the same architecture initialised with ImageNet weights. Both models use the same 500 species, data splits, transforms and evaluation code.

The image data are not committed to GitHub. At present the expected layout is:

```text
project root/
  train_mini/   # 500 class folders, 50 images per class
  val/          # the held-out test set, 10 images per class
```

`train_mini` is split reproducibly into 40 training and 10 validation images per species. The official iNaturalist validation images in `val` are used only for the final test evaluation.

## Repository layout

```text
configs/        experiment settings
metadata/       generated split manifests and class-index mapping
notebooks/      training, evaluation and Grad-CAM notebooks
src/            code shared between experiments
results/        small result tables and selected figures
checkpoints/    local checkpoints (ignored by Git)
tests/          small checks for shared code
```

The training loop deliberately remains in the notebooks so that the experimental procedure is easy to follow. Dataset handling, model construction, checkpointing, metrics and Grad-CAM are in `src/inat_project` so later experiments do not have to duplicate them.

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open `notebooks/01_resnet18_scratch.ipynb` for the random-initialisation experiment. The first setup cell contains the only machine-specific paths. On Colab, extract the dataset to the runtime's `/content` disk and save checkpoints to Google Drive.

## Reproducibility

- The selected 500 class-folder names determine the label set.
- `metadata/class_to_idx.json` is the canonical label mapping.
- Split manifests are generated with seed 42 and should be committed.
- Training and validation come from `train_mini`; `val` is never used for model selection.
- Checkpoints record the configuration, optimiser, scheduler, epoch and label mapping.

The two known duplicate pairs in `train_mini` are kept together in the training portion by the manifest builder. This avoids identical content appearing in both the training and validation partitions.

## Results expected from each deep model

- training and validation loss/top-1 curves
- top-1 and top-5 test accuracy
- macro precision, recall and F1
- balanced accuracy
- confusion matrix and most common off-diagonal confusions
- training and inference time
- per-image prediction CSV

Large model files and the image dataset must not be added to the submitted source-code archive.

