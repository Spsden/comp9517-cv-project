# COMP9517 iNaturalist species classification

This is our COMP9517 group project. We compare handcrafted image features, a
ResNet-18 trained from scratch, and a ResNet-18 fine-tuned from ImageNet
weights.

To run it on your computer, point a private `.env` file to your copy of
`selected_images.tar` or `selected_images.tar.gz`. The setup command will
extract it into a Git-ignored folder and check that the shared dataset splits
are ready. You do not need to download the full iNaturalist dataset.

## Local quick start

Use Python 3.10 or later. Open a terminal in the repository folder and run:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python -m ipykernel install --user --name comp9517-inat --display-name "Python (COMP9517 iNaturalist)"

cp .env.example .env
# Edit .env and set INAT_DATA_ARCHIVE to your local selected_images.tar(.gz).

inat-prepare
jupyter lab
```

Before you run a notebook, select **Kernel > Change Kernel > Python (COMP9517
iNaturalist)**. Please check this carefully: Jupyter must use the same `.venv`
where you installed the project. You can confirm it in any notebook cell with:

```python
import sys
print(sys.executable)
```

The path should end in `.venv/bin/python` on macOS/Linux or
`.venv\\Scripts\\python.exe` on Windows. If you see a different Python path,
switch the kernel before running the rest of the notebook.

You normally need to run `inat-prepare` only once. It will:

1. Read `.env` (your real file is ignored by Git).
2. Safely extract the archive to `data/inat500/` by default.
3. Check that the archive contains `train_mini/` and `val/`.
4. Check the 500-class, 40/10/10 data splits. Only use `inat-prepare --rebuild-manifests` if you intentionally want to recreate them.

After that, your local files should look like this:

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

We kept the training loops in the notebooks so you can follow each experiment.
The reusable parts--dataset loading, paths, archive extraction, model creation,
checkpoints, metrics, and Grad-CAM--live in `src/inat_project`.

## Notebooks

Once `inat-prepare` finishes successfully, open `notebooks/` and run these in
order:

1. `01_resnet18_scratch.ipynb` - random-initialisation baseline.
2. `02_resnet18_pretrained.ipynb` - ImageNet fine-tuning.
3. `03_gradcam_analysis.ipynb` - Grad-CAM for a completed checkpoint.
4. `04_model_comparison.ipynb` - consolidated deep-model analysis; it only needs result files.
5. `05_hog_hsv_classical.ipynb` and `05_traditional_svm_sift.ipynb` - two independent traditional baselines.

Every notebook uses the same `.env` configuration. If it cannot find the
dataset, the first setup cell will tell you to run `inat-prepare`. None of the
notebooks downloads data or mounts Google Drive.

`notebooks/05_hog_hsv_classical.ipynb` contains the HOG, HSV, and combined
HOG+HSV linear-SVM baselines. It uses exactly the same splits as the deep
models and chooses the SVM regularisation value using validation macro F1.

## Please keep these experiment settings the same

- Use the selected 500 class folders as the label set.
- Keep `metadata/class_to_idx.json` as the shared label mapping.
- Keep the split seed at 42 and commit the split files.
- Use `train_mini` for training and validation. Do not use `val` while choosing a model.
- Keep the configuration, optimiser, scheduler, epoch, and label mapping in each checkpoint.

The two known duplicate pairs in `train_mini` are kept together in the training portion by the manifest builder. This avoids identical content appearing in both the training and validation partitions.

## What each deep model should produce

- training and validation loss/top-1 curves
- top-1 and top-5 test accuracy
- macro precision, recall and F1
- balanced accuracy
- confusion matrix and most common off-diagonal confusions
- training and inference time
- per-image prediction CSV

## Path configuration

For your first setup, the only required setting is `INAT_DATA_ARCHIVE`. It can
be an absolute path or a path relative to the repository folder. The other
settings in [.env.example](.env.example) are optional. They are useful if, for
example, you want to keep the extracted data on an external drive:

```dotenv
INAT_DATA_ARCHIVE=/Volumes/external-drive/selected_images.tar.gz
INAT_DATA_DIR=/Volumes/external-drive/inat500
```

If you are unsure which paths the project is using, run
`inat-prepare --show-paths`. Values set in your shell override `.env`, which is
handy for temporary runs or CI.

Please do not add model weights, image archives, extracted images, cache files,
or your real `.env` file to Git.
