# CLIP-GPT2 Vision-Language Model

A quick research project combining CLIP and pre-trained GPT-2 to generate image descriptions without training large models to determine if a VLM can be done without training.

**Author**: Eduardo J. Barrios ([@edujbarruos](https://github.com/edujbarruos))

## Overview

This project explores creating a vision-language model by:
- Using CLIP to encode images into embeddings
- Mapping CLIP embeddings to GPT-2's input space
- Using pre-trained GPT-2 to generate natural language descriptions

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from ejb_vlm_model import CLIPGPTDescriptor

# Initialize the model
model = CLIPGPTDescriptor()

# Generate description for an image
description = model.describe_image("path/to/image.jpg")
print(description)
```

## Project Structure

- `ejb_vlm_model.py` - Main vision-language model implementation (EJB = Eduardo J. Barrios)
- `example_usage.py` - Example scripts and demonstrations
- `utils.py` - Helper functions
- `requirements.txt` - Project dependencies

## Research Goal

Testing the feasibility of creating a VLM without training large models from scratch, leveraging pre-trained components.
