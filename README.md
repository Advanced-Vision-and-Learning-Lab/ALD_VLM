# Parameter-efficient Vision-Language Modeling for Alcoholic Liver Disease Histology

## Overview
This project introduces an efficient pipeline for classifying and analyzing Whole Slide Images (WSIs) of Alcoholic Liver Disease (ALD), leveraging domain-specific foundational Vision-Language Models (VLMs) like CONCH. 

## Pipeline 

# Step 1: Extract tissue patches
python prepare_dataset.py

# Step 2: Encode patches using the Vision-Language Model
python extract_features.py --model CONCH

# Step 3: Train the MIL Adapter
python train.py --adapter CLIPAdapter --encoder CONCH --aggregator ABMIL

# Step 4: Evaluate the model and heatmaps
python inference.py --adapter CLIPAdapter --encoder CONCH --aggregator ABMIL --generate_visuals

## Installation Prerequisites

The [`requirements.txt`](requirements.txt) file includes the necessary packages, and the packages will be installed using:

   ```pip install -r requirements.txt```

## Project Structure
```text
├── dataset/
│   ├── raw_wsis/                  # Place your raw .svs slide files here
│   └── prepared_dataset/          # Auto-generated patch coordinates and tensors
│       ├── index.csv
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   ├── models.py                  # Core PyTorch architectures (ABMIL, Adapters)
│   ├── CONCH/                     # Local directory for CONCH weights (pytorch_model.bin)
│   └── cache_for_HF/              # Offline HuggingFace cache
├── prepare_dataset.py             # Tissue masking and patching
├── extract_features.py            # VLM encoding (Text & Image)
├── train.py                       # Model training loop
└── inference.py                   # Evaluation and visualization
