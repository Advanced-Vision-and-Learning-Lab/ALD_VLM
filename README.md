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

