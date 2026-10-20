# Parameter-Efficient Vision–Language Modeling for Liver Histology

Few-shot classification of drinking status from rhesus macaque liver whole-slide
images in the Monkey Alcohol Tissue Research Resource (MATRR). The pipeline uses
frozen CONCH image and text encoders, multiple instance learning (MIL), and
optional visual adapters or text residuals. The prediction target is drinking
status; it is not a histological diagnosis of alcohol-associated liver disease.

## Installation

```bash
pip install -r requirements.txt
```

CONCH weights are required at `models/CONCH/pytorch_model.bin`. A CUDA-capable GPU
is recommended for whole-slide feature extraction. `environment.yml` records the
Conda environment used for development.

## Project structure

```text
demo.py                         # Feature extraction, training, and evaluation
utils/                          # Models, losses, data loading, and visualization
dataset/raw_wsis/               # Input .svs whole-slide images
models/CONCH/pytorch_model.bin  # CONCH checkpoint
models/cache_for_HF/            # Hugging Face model and tokenizer cache
outputs/                       # Features, subject splits, and experiment results
```

The subject metadata in `demo.py` specifies the 37-animal cohort. Slide filenames
must include the cohort and subject ID, for example
`Coh7a_10091_Nx_Liver_HE.svs`. Preprocessing selects the left half of each slide,
filters tissue patches, and extracts 512-dimensional CONCH features. Physical
patches are 224 × 224 pixels at level 0; the encoder's supplied transform resizes
them to 448 × 448. Text inputs use CONCH's 128-position tokenization.

## Training

For a 4-shot experiment with an adapter bottleneck of 128 and LSE text pooling:

```bash
python demo.py --model abmil_adapter --reduction 4 --text_pool lse \
    --shots 4 --iterations 10 --seeds 3 --epochs 250 --lr 1e-3 --patience 50
```

Use `--reduction 64` for an adapter bottleneck of 8. The MIL attention dimension
is controlled separately by `--abmil_d` and defaults to 64. Model choices include
Max Pool (`max_pool`), ABMIL (`zsmil`), ABMIL Adapter (`abmil_adapter`), and
TaskRes (`taskres`). Text prompts support mean and log-sum-exp (`lse`) pooling.

To add supervised contrastive loss, append:

```text
--supcon_weight 0.1 --supcon_temperature 0.1
```

The objective is classification loss plus the weighted contrastive loss. Each
training animal contributes one slide representation; same-class animals form
positive pairs. Validation and test animals are excluded from this loss. No
projection network or additional trainable parameters are introduced. This option
requires at least two training animals per class and a trainable visual
representation; frozen Max Pool without a visual adapter is unsupported. Joint
contrastive training can require more GPU memory than classification-only
training. Early stopping selects the lowest validation classification loss.

## Evaluation and outputs

Each split assigns distinct animals to training, validation, and testing, with
the requested number of animals per class in both training and validation.
For this cohort, 2-shot uses 4 training, 4 validation, and 29 test animals;
4-shot uses 8 training, 8 validation, and 21 test animals. Repeated splits reuse
animals and do not constitute independent cohorts.

New experiments default to `outputs/experiments/`; `--results_root` selects
another directory. Classification-only results are stored under `raw/`, and
contrastive results under `contrastive/`. Each experiment records its settings,
source hashes, metrics, predictions, checkpoints, learning curves, and ROC curves.
Attention-based models also produce whole-slide visualizations unless
`--skip_xai` is specified.

Feature caches are stored in `outputs/features_conch_v2/` with extraction
metadata. Saved subject splits are reused across methods. Configuration checks
prevent incompatible experiments from overwriting existing results.

Reported metrics are balanced accuracy, ROC AUC, and positive-class F1. Run-level
standard deviations describe variability across overlapping splits and seeds;
they are not confidence intervals over independent animals. Attention maps
visualize model aggregation weights and require separate validation for
histological interpretation.
