# Prompt Engineering Optimization

This repository contains code for optimizing text-to-image prompts using Direct Preference Optimization (DPO) and other techniques.

## Overview

The project compares different models for text-to-image prompt enhancement:
- SFT (Supervised Fine-Tuning) baseline
- Promptist
- Bloom-based models
- DPO-optimized models with various hyperparameters

Evaluation metrics include:
- Human Preference Score (HPS)
- CLIP score
- Aesthetic scores
- ImageReward metrics

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required model weights:
```bash
# Example commands for downloading model weights if applicable
```

3. Prepare the dataset:
- Place your prompt CSV file in the root directory

## Usage

Run the evaluation:
```bash
python dpop_exp.py
```

Results will be saved to `diffusion_output.csv`.

## Models

- GPT-2 based models fine-tuned on prompt optimization
- Models with varying DPO parameters (0.1-0.9)
- Promptist from Microsoft
- Bloom-based text-to-prompt models 