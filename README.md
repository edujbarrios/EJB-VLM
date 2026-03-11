# EJB-VLM

**Eduardo J. Barrios Vision-Language Model**

A quick research project combining CLIP and pre-trained GPT-2 to generate image descriptions without training large models. Testing if a functional VLM can be implemented using only pre-trained components.

**Author**: Eduardo J. Barrios ([@edujbarrIos](https://github.com/edujbarrIos))

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

### Run the Demo

```bash
python examples/demo.py
```

This will download a sample image and demonstrate the model's capabilities.

### Basic Usage

```python
from src.models.ejb_vlm_model import CLIPGPTDescriptor

# Initialize the model (loads config automatically)
model = CLIPGPTDescriptor()

# Generate description for an image
description = model.describe_image("path/to/image.jpg")
print(description)
```

### Advanced Usage

```python
from src.models.ejb_vlm_model import AdvancedCLIPGPTDescriptor

# Initialize advanced model with extended categories
model = AdvancedCLIPGPTDescriptor(category_set="extended")

# Get detailed analysis
result = model.detailed_description("image.jpg", num_descriptions=3)

print("Categories:", result["categories"])
print("Descriptions:", result["descriptions"])

# Use different generation presets
creative_desc = model.describe_image("image.jpg", preset="creative")
detailed_desc = model.describe_image("image.jpg", preset="detailed")
```

### Interactive Mode

```bash
python examples/example_usage.py interactive
```

## Project Structure

```
ejb-vlm/
├── config/
│   └── config.yaml          # Centralized configuration
├── src/
│   ├── models/
│   │   └── ejb_vlm_model.py # Main VLM implementation
│   └── utils/
│       ├── config_loader.py # Configuration management
│       ├── image_utils.py   # Image processing utilities
│       ├── device_utils.py  # Device detection
│       └── io_utils.py      # I/O operations
├── examples/
│   ├── demo.py              # Quick demo script
│   └── example_usage.py     # Comprehensive examples
├── data/                    # Data directory
├── tests/                   # Test files (coming soon)
└── requirements.txt         # Dependencies
```

## Features

### Core Capabilities
- ✅ Zero-shot image captioning using CLIP + GPT-2
- ✅ Parameterized configuration via YAML
- ✅ Multiple generation presets (default, creative, focused, detailed)
- ✅ Category detection using CLIP's zero-shot classification
- ✅ Image similarity comparison
- ✅ Batch processing of multiple images
- ✅ Modular architecture for easy extension

### Models Used
- **CLIP (ViT-B/32)**: Image encoding and zero-shot classification
- **GPT-2**: Natural language generation

## Research Goal

Testing the feasibility of creating a VLM without training large models from scratch, leveraging pre-trained components. This approach explores whether combining existing models can produce meaningful image descriptions without expensive fine-tuning.

## Examples

See various usage modes:
```bash
python examples/example_usage.py basic       # Single image description
python examples/example_usage.py advanced    # With category detection
python examples/example_usage.py presets     # Compare generation presets
python examples/example_usage.py batch       # Process multiple images
python examples/example_usage.py compare     # Compare image similarity
```

## Configuration

All parameters are centralized in `config/config.yaml`:

- **Models**: CLIP and GPT-2 model variants
- **Generation Presets**: Default, creative, focused, detailed
- **Categories**: Default and extended category sets
- **Paths**: Data, output, cache directories
- **Demo Settings**: Sample images and display options

Modify `config/config.yaml` to customize behavior without changing code.

## Why "EJB-VLM"?

- **EJB**: Eduardo J. Barrios - emphasizing authorship
- **VLM**: Vision-Language Model - describes the core functionality

This is a research project exploring whether functional VLMs can be built without training large models, using only pre-trained components and smart integration.

---

**Research Question**: Can we create a working vision-language model by combining pre-trained CLIP and GPT-2 without any training?

**Answer**: Yes! This project demonstrates that meaningful image descriptions can be generated through zero-shot prompting techniques.
