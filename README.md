# Parameter-efficient Vision-Language Modeling for Alcoholic Liver Disease Histology

## Overview
This project introduces an efficient pipeline for classifying and analyzing Whole Slide Images (WSIs) of Alcoholic Liver Disease (ALD), leveraging domain-specific foundational Vision-Language Models (VLMs) like CONCH. 

## Installation Prerequisites

The [`requirements.txt`](requirements.txt) file includes the necessary packages, and the packages will be installed using:

   ```pip install -r requirements.txt```

## Project Structure
```text
├── dataset/
│   ├── raw_wsis/                  # Place your raw .svs slide files here
├── models/
│   ├── models.py                  # Core PyTorch architectures (ABMIL, Adapters)
│   ├── CONCH/                     # Local directory for CONCH weights (pytorch_model.bin)
│   └── cache_for_HF/              # Offline HuggingFace cache
```

## Corrected experiment pipeline

See [the current experiment guide](docs/experiment_guide.md) for commands,
augmentation, and what is still unresolved about the low AUC/F1.
The [original research/code audit](docs/research_audit.md) records the earlier review.
The corrected pipeline uses CONCH's supplied
448-pixel image transform and 128-position text input contract. Legacy cached features
and paper results are preserved; corrected features go to `outputs/features_conch_v2`
and results to `outputs/revalidated/raw` or `outputs/revalidated/augmented`.

With conda available (the runner activates `ald-vlm-env`):

```bash
bash scripts/run_revalidated.sh
```

This runs tests, then **2-shot and 4-shot only**, both without and with training-only
image augmentation, including the original models and an ABMIL Adapter variant with
attention size 8 and adapter size 8 (16,426 trainable parameters).
The two logistic baselines are excluded from the runner and reports.
There are 48 configurations / 1,440 fitted runs, using ten saved splits and three seeds.
Completed experiments are skipped before launching Python and their logs are preserved.
The runner generates both tables and figures. A GPU is expected for full extraction. For fast
verification in the environment:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/check_conch_inputs.py  # One real patch; needs local slide and weights
```

After experiments complete, both reporting scripts work without arguments:

```bash
python generate_latex_table.py
python generate_fig.py
```

The tables are `outputs/revalidated/classification_table.tex` and
`outputs/revalidated/classification_table_augmented.tex`; both contain only 2/4-shot
columns and 12 method/pooling rows, including the smaller model. An individual run enables augmentation with `--augment`
(`--aug_views 2 --aug_seed 42` by default). Validation/test images stay unaugmented.
The shell script automatically runs both conditions. The left-half WSI selection is preserved.

Use a different `RESULTS_ROOT` for changed experiment configurations; matching completed
runs can be reused. `ALD_RESULTS_ROOT` selects that directory for standalone figure/table
generation. Source images, coordinates, and historical results are not overwritten.

