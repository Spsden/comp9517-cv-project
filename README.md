# COMP9517 iNaturalist species classification

This repository contains our COMP9517 group project: a comparison between handcrafted features, a ResNet-18 trained from random initialisation, and the same architecture fine-tuned from ImageNet weights.  The project is now designed to run locally after a clone; no Colab, Google Drive, or machine-specific paths are required.

The image data are deliberately excluded from Git.  Each developer points a private `.env` file to their local copy of the shared `selected_images.tar` or `selected_images.tar.gz` archive.  The setup command extracts it safely into a Git-ignored location and validates the shared manifests.

## Local quick start

Python 3.10 or later is required.  From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .

cp .env.example .env
# Edit .env and set INAT_DATA_ARCHIVE to your local selected_images.tar(.gz).

inat-prepare
jupyter lab
```

`inat-prepare` does the following once per machine:

1. Reads `.env` (the real file is ignored by Git).
2. Safely extracts the archive to `data/inat500/` by default.
3. Requires the extracted archive to contain `train_mini/` and `val/`.
4. Validates the 500-class, 40/10/10 split manifests; use `inat-prepare --rebuild-manifests` only when deliberately recreating them.

The resulting local layout is:

```text
project root/
  .env                         # private path configuration; never commit
  data/inat500/
    train_mini/                # 500 class folders, 50 images per class
    val/                       # held-out test images, 10 images per class
  metadata/                    # canonical split manifests and class mapping
  checkpoints/                 # local, Git-ignored model weights
  results/                     # experiment metrics, figures, predictions
```

`train_mini` is split reproducibly into 40 training and 10 validation images per species. The official iNaturalist validation images in `val` are used only for final test evaluation.

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

The training loops deliberately remain in notebooks so the experimental procedure is easy to follow. Dataset handling, portable path configuration, safe archive extraction, model construction, checkpointing, metrics, and Grad-CAM are shared under `src/inat_project`.

## Notebooks

After `inat-prepare` succeeds, open and run the notebooks from `notebooks/` in this order:

1. `01_resnet18_scratch.ipynb` - random-initialisation baseline.
2. `02_resnet18_pretrained.ipynb` - ImageNet fine-tuning.
3. `03_gradcam_analysis.ipynb` - Grad-CAM for a completed checkpoint.
4. `04_model_comparison.ipynb` - consolidated deep-model analysis; it only needs result files.
5. `05_hog_hsv_classical.ipynb` and `05_traditional_svm_sift.ipynb` - two independent traditional baselines.

The notebooks read the same local path configuration.  If the dataset is missing, their first setup cell tells you to run `inat-prepare`; they never download data or mount Drive.

`00_prepare_dataset.ipynb` documents the original one-off process used to make the shared 500-class archive from the full iNaturalist release.  It is not part of normal local setup when you already have `selected_images.tar(.gz)`.

`notebooks/05_hog_hsv_classical.ipynb` contains the HOG, HSV and combined HOG+HSV linear-SVM baselines. It uses the same manifest-backed splits as the deep models and selects the SVM regularisation parameter using validation macro F1.

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

## Path configuration

Only `INAT_DATA_ARCHIVE` is required for a new local setup.  It can be an absolute path or a path relative to the repository root.  The remaining variables shown in [.env.example](.env.example) are optional overrides if, for example, you keep data on an external drive:

```dotenv
INAT_DATA_ARCHIVE=/Volumes/external-drive/selected_images.tar.gz
INAT_DATA_DIR=/Volumes/external-drive/inat500
```

Run `inat-prepare --show-paths` to check what the current configuration resolves to.  Environment variables supplied by your shell override `.env`, which is useful for CI or temporary runs.

Large model files, image archives, extracted images, cache files, and real `.env` files must not be added to Git.
