# Working agreement

The shared pieces of the experiment are intentionally small. Please use them rather than copying a second version into another notebook.

## Before starting an experiment

1. Pull the current repository and create a short-lived branch.
2. Do not regenerate `metadata/splits` with another seed.
3. Check that `metadata/class_to_idx.json` matches the mapping stored in any checkpoint you load.
4. Add a separate YAML file under `configs` when changing an experimental condition.

The scratch and pretrained ResNet-18 experiments must use the same manifests, image transforms and test evaluation. If a shared component genuinely needs to change, discuss it first and rerun every affected comparison.

## Notebooks

Use the numbering below so the intended order remains clear:

```text
01_resnet18_scratch.ipynb
02_resnet18_pretrained.ipynb
03_gradcam_analysis.ipynb
04_model_comparison.ipynb
```

Keep notebook outputs small before committing. A notebook should run in order from a fresh runtime, apart from the documented path settings and long training cell.

## Results

Each experiment gets its own folder under `results`. Commit JSON/CSV summaries that are needed for the group comparison. Keep checkpoints and bulk image outputs in Drive. Copy only selected final figures into the report or presentation folders.

## Pull requests

A useful pull request states:

- what experimental question was tested;
- which configuration file was used;
- whether the data manifests or shared evaluation changed;
- the best validation epoch and final test metrics;
- how to reproduce the result.

