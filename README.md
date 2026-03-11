# MedGAN - Medical Image Description Generator

**Author:** Eduardo J. Barrios (@edujbarrios)

A quick research project combining MedClip and pre-trained language models to generate descriptions for medical images without additional training.

## Overview

**Research Goal:** This project investigates whether a functional vision-language model for medical images can be constructed using only pre-trained components (MedClip + GAN/RNN) without training larger models from scratch.

This project uses:
- **MedClip**: Pre-trained medical vision-language model for image encoding
- **Pre-trained Language Model**: GPT-2 or similar RNN/GAN-based text generator
- **Zero-shot approach**: No additional training required - leveraging existing model capabilities

## Features

- Medical image encoding using MedClip
- Automatic caption generation
- Simple inference pipeline
- Easy to extend and experiment

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from medgan_pipeline import MedicalImageDescriptor

# Initialize the pipeline
descriptor = MedicalImageDescriptor()

# Generate description for an image
description = descriptor.describe_image("path/to/medical_image.jpg")
print(description)
```

## Project Structure

- `medgan_pipeline.py`: Main pipeline implementation
- `medclip_encoder.py`: MedClip image encoding module
- `text_generator.py`: Text generation module
- `requirements.txt`: Project dependencies
- `examples/`: Example usage scripts

## Research Notes

**Research Question:** Can a vision-language model for medical imaging be effectively created by combining pre-trained MedClip embeddings with a pre-trained GAN/RNN text generator, thereby avoiding the need to train large, computationally expensive models?

This experimental project explores:
- Feasibility of zero-shot medical image description
- Quality of descriptions generated without domain-specific fine-tuning
- Computational efficiency compared to training full vision-language models
- Potential for rapid prototyping of medical AI applications

**Hypothesis:** By leveraging MedClip's medical domain knowledge and a language model's text generation capabilities, we can create a functional vision-language system without additional training overhead.
