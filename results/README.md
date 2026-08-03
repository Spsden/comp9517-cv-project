# Results

Store one compact result bundle per experiment:

```text
results/<experiment>/
  metrics.json
  runtime.json
  training_history.csv
  test_predictions.csv
  confused_pairs.csv
  figures/
```

Checkpoints do not belong here. They are large, machine-specific working files and are saved in Google Drive during Colab training.

