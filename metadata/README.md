# Dataset metadata

These files fix the data partition used by every group member.

- `class_to_idx.json`: canonical mapping from the 500 folder names to output indices 0–499.
- `split_summary.json`: split seed and expected totals.
- `splits/train.csv`: 20,000 images from `train_mini`.
- `splits/validation.csv`: 5,000 different images from `train_mini`.
- `splits/test.csv`: 5,000 images from the official iNaturalist validation subset.

The manifests were generated with seed 42. Paths are relative to the dataset root, so the same files work on macOS and Colab. Do not edit rows manually or regenerate them for individual models.

