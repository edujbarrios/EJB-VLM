# EJB-VLM

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A quick research project combining CLIP and pre-trained GPT-2 to generate image descriptions without training large models. Testing if a functional VLM can be implemented using only pre-trained components.

**Author**: Eduardo J. Barrios ([@edujbarrIos](https://github.com/edujbarrIos))

[Features](#features) • [Quick Start](#quick-start) • [Examples](#examples) • [Medical Use](#medical-imaging-use-cases) • [Documentation](#api-reference)

</div>

## 🆕 What's New

### Medical Imaging Support with MedCLIP
Now includes support for **MedCLIP** - a specialized CLIP variant trained on medical images (PubMed, MIMIC-CXR datasets). This enables:
- Enhanced analysis of chest X-rays, CT scans, and MRI images
- Better understanding of histopathology and microscopy images
- Specialized medical terminology and clinical context
- Improved zero-shot classification for radiological images

Perfect for researchers and clinicians working with medical imaging data!

## Overview

This project explores creating a vision-language model by:
- Using CLIP to encode images into embeddings
- Mapping CLIP embeddings to GPT-2's input space
- Using pre-trained GPT-2 to generate natural language descriptions

**New**: Supports both standard CLIP and MedCLIP for domain-specific applications.

## Requirements

```bash
pip install -r requirements.txt
```

### Additional Notes
- **MedCLIP**: Automatically downloaded from Hugging Face on first use
- **GPU Recommended**: For faster inference, especially with medical images
- **Memory**: MedCLIP requires ~2GB GPU memory (or runs on CPU)

## Quick Start

### Run the Demo

```bash
python examples/demo.py
```

This will download a sample image and demonstrate the model's capabilities.

### Basic Usage

```python
from src.models.ejb_vlm_model import EJBVLMDescriptor

# Initialize the model (loads config automatically)
model = EJBVLMDescriptor()

# Generate description for an image
description = model.describe_image("path/to/image.jpg")
print(description)
```

### Medical Imaging Usage (NEW!)

```python
from src.models.ejb_vlm_model import EJBVLMDescriptor

# Initialize with MedCLIP for medical images
medical_model = EJBVLMDescriptor(
    clip_model_name="flaviagiammarino/pubmed-clip-vit-base-patch32"
)

# Analyze a chest X-ray
description = medical_model.describe_image("chest_xray.jpg")
print(description)

# Or use the AdvancedEJBVLMDescriptor with medical categories
from src.models.ejb_vlm_model import AdvancedEJBVLMDescriptor

medical_advanced = AdvancedEJBVLMDescriptor(
    clip_model_name="flaviagiammarino/pubmed-clip-vit-base-patch32",
    category_set="medical"
)

result = medical_advanced.detailed_description("ct_scan.jpg")
print("Detected modality:", result["categories"])
print("Clinical description:", result["descriptions"])
```

### Advanced Usage

```python
from src.models.ejb_vlm_model import AdvancedEJBVLMDescriptor

# Initialize advanced model with extended categories
model = AdvancedEJBVLMDescriptor(category_set="extended")

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


## Features

### Core Capabilities
- ✅ Zero-shot image captioning using CLIP + GPT-2
- ✅ **NEW: MedCLIP support for medical imaging analysis**
- ✅ Parameterized configuration via YAML
- ✅ **Multiple model variants (Standard, Medical)**
- ✅ Multiple generation presets (default, creative, focused, detailed)
- ✅ Category detection using CLIP's zero-shot classification
- ✅ **Domain-specific categories (General, Extended, Medical)**
- ✅ Image similarity comparison
- ✅ Batch processing of multiple images
- ✅ Modular architecture for easy extension

### Models Used
- **CLIP (ViT-B/32)**: Image encoding and zero-shot classification
- **MedCLIP**: Specialized CLIP for medical images (PubMed, MIMIC-CXR trained)
- **GPT-2**: Natural language generation

## Research Goal

Testing the feasibility of creating a VLM without training large models from scratch, leveraging pre-trained components. This approach explores whether combining existing models can produce meaningful image descriptions without expensive fine-tuning.

## Examples

See various usage modes:
```bash
python examples/demo.py                      # Quick demo with sample image
python examples/example_usage.py basic       # Single image description
python examples/example_usage.py advanced    # With category detection
python examples/example_usage.py presets     # Compare generation presets
python examples/example_usage.py batch       # Process multiple images
python examples/example_usage.py compare     # Compare image similarity
python examples/demo_medical.py              # Medical imaging with MedCLIP
python examples/demo_variants.py             # Switching between model variants
python examples/advanced_examples.py         # Image similarity & clustering
python examples/batch_processing.py          # Efficient batch operations
python examples/interactive_cli.py           # Interactive command-line interface
```

Or install and use the CLI:
```bash
pip install -e .
ejb-vlm  # Launch interactive mode
```

## Configuration

All parameters are centralized in `config/config.yaml`:

- **Models**: CLIP and GPT-2 model variants
- **Variants**: Pre-configured setups (Standard, Medical)
- **Generation Presets**: Default, creative, focused, detailed
- **Categories**: Default, extended, and medical category sets
- **Templates**: Basic, extended, and medical context templates
- **Paths**: Data, output, cache directories
- **Demo Settings**: Sample images and display options

Modify `config/config.yaml` to customize behavior without changing code.

### Available Model Variants

1. **Standard** (default): General-purpose VLM for everyday images
   - Uses: CLIP ViT-B/32
   - Categories: General objects, scenes, activities
   
2. **Medical** (NEW!): Specialized for clinical and biomedical images
   - Uses: MedCLIP (PubMed-trained)
   - Categories: X-rays, CT, MRI, histopathology, microscopy
   - Templates: Clinical and diagnostic terminology

## Medical Imaging Use Cases

The MedCLIP variant is particularly useful for:

- **Radiology**: Analyzing chest X-rays, CT scans, MRI images
- **Pathology**: Describing histopathology slides and microscopy images  
- **Research**: Automating image annotation for medical datasets
- **Education**: Generating descriptions for teaching materials
- **Clinical Documentation**: Assisting with preliminary image interpretation

### Medical Disclaimer

⚠️ **Important**: This tool is designed for research and educational purposes only. It is **NOT** intended for clinical diagnosis, treatment decisions, or medical advice. Always consult qualified healthcare professionals for medical interpretation and decisions.

## API Reference

### EJBVLMDescriptor

Basic vision-language model for image description.

```python
EJBVLMDescriptor(
    config=None,              # Config dict (loads from yaml if None)
    clip_model_name=None,     # CLIP variant (e.g., "ViT-B/32")
    gpt_model_name=None,      # GPT-2 variant (e.g., "gpt2-medium")
    device=None               # Device: "cuda", "cpu", or "auto"
)
```

**Key Methods:**

- `describe_image(image_path, preset=None, **kwargs)` - Generate description
- `encode_image(image_path)` - Get CLIP embeddings
- `batch_describe_images(image_paths, **kwargs)` - Process multiple images

### AdvancedEJBVLMDescriptor

Extended model with category detection and advanced features.

```python
AdvancedEJBVLMDescriptor(
    config=None,
    clip_model_name=None,
    gpt_model_name=None,
    device=None,
    category_set="default"    # "default", "extended", or "medical"
)
```

**Key Methods:**

- `detailed_description(image_path, num_descriptions=1, top_categories=3)` - Complete analysis
- `detect_categories(image_path, top_k=5)` - Zero-shot classification
- `compare_images(image_path1, image_path2)` - Similarity scoring

### Generation Parameters

All generation methods accept these parameters:

```python
description = model.describe_image(
    "image.jpg",
    preset="creative",        # Optional: "default", "creative", "focused", "detailed"
    max_length=60,            # Maximum tokens
    temperature=0.8,          # Randomness (0.0-2.0)
    top_p=0.95,               # Nucleus sampling
    top_k=50,                 # Top-k sampling
    num_beams=5,              # Beam search width
    num_return_sequences=3    # Number of outputs
)
```

## Troubleshooting

### Common Issues

**Problem**: Out of memory error
```python
# Solution: Use CPU or smaller model
model = EJBVLMDescriptor(device="cpu")
# Or use smaller CLIP variant
model = EJBVLMDescriptor(clip_model_name="RN50")
```

**Problem**: Slow inference
```python
# Solution: Reduce beam search
description = model.describe_image("image.jpg", num_beams=3)
# Or use faster preset
description = model.describe_image("image.jpg", preset="creative")
```

**Problem**: Generic descriptions
```python
# Solution: Use detailed preset or adjust temperature
description = model.describe_image("image.jpg", preset="detailed")
# Or increase creativity
description = model.describe_image("image.jpg", temperature=1.2)
```

**Problem**: MedCLIP not loading
```bash
# Solution: Ensure you have internet connection for first download
# Or manually download from Hugging Face
# https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32
```

### Performance Tips

1. **GPU Acceleration**: Always use CUDA when available
   ```python
   model = EJBVLMDescriptor(device="cuda")
   ```

2. **Batch Processing**: Use batch methods for multiple images
   ```python
   results = model.batch_describe_images(["img1.jpg", "img2.jpg"])
   ```

3. **Model Caching**: Models are cached after first load (faster subsequent runs)

4. **Image Preprocessing**: Resize large images before processing
   ```python
   from PIL import Image
   img = Image.open("large.jpg")
   img.thumbnail((800, 800))
   img.save("resized.jpg")
   ```

## Roadmap & Future Enhancements

### In Progress
- [ ] 🔬 Additional medical imaging datasets for evaluation
- [ ] 📊 Quantitative benchmarking suite
- [ ] 🌐 Multi-language support (multilingual CLIP)

### Planned Features
- [ ] 🎨 Art-specialized variant (ArtCLIP)
- [ ] 🛰️ Satellite imagery variant (RemoteCLIP)
- [ ] 📹 Video description support
- [ ] 🔌 REST API wrapper
- [ ] 🖼️ Gradio web interface
- [ ] 📦 Docker container
- [ ] ⚡ ONNX export for faster inference
- [ ] 🔄 Fine-tuning scripts for custom domains

### Research Directions
- [ ] Hybrid models (CLIP + LLaMA/Mistral)
- [ ] Attention visualization
- [ ] Prompt engineering optimization
- [ ] Zero-shot medical report generation

## Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue with details and reproduction steps
2. **Suggest Features**: Share ideas for new capabilities or improvements
3. **Add Model Variants**: Implement support for new CLIP variants
4. **Improve Documentation**: Fix typos, add examples, clarify instructions
5. **Submit Code**: Fork, create a branch, and submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/edujbarrIos/ejb-vlm.git
cd ejb-vlm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests (coming soon)
python -m pytest tests/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions
- Keep configuration in YAML files

## Citation

If you use this project in your research, please cite:

```bibtex
@software{barrios2026ejbvlm,
  author = {Barrios, Eduardo J.},
  title = {EJB-VLM: Zero-Shot Vision-Language Model using CLIP and GPT-2},
  year = {2026},
  url = {https://github.com/edujbarrIos/ejb-vlm},
  note = {Research project exploring parametrized VLM architectures}
}
```

### Related Work

- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **MedCLIP**: [MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://arxiv.org/abs/2210.10163)
- **GPT-2**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Documentation

- **[README.md](README.md)** - Main documentation (you are here)
- **[FAQ.md](FAQ.md)** - Frequently asked questions
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[LICENSE](LICENSE)** - MIT License details

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for CLIP and GPT-2 models
- Hugging Face for model hosting and transformers library
- The medical imaging community for MedCLIP development
- All contributors and users of this project

## Support

- 📖 **Documentation**: Check [README.md](#readme), [FAQ.md](FAQ.md), and [Examples](#examples)
- 🐛 **Bug Reports**: [Open an issue](https://github.com/edujbarrIos/ejb-vlm/issues/new?template=bug_report.md)
- 💡 **Feature Requests**: [Suggest a feature](https://github.com/edujbarrIos/ejb-vlm/issues/new?template=feature_request.md)
- 🤖 **Model Variants**: [Suggest a variant](https://github.com/edujbarrIos/ejb-vlm/issues/new?template=model_variant.md)
- 💬 **Questions**: Open a discussion or issue

---

**Research Questions**: 
1. Can we create a working vision-language model by combining pre-trained CLIP and GPT-2 without any training?
2. Can domain-specific CLIP models (like MedCLIP) improve performance on specialized image types?

**Answers**: 
1. ✅ Yes! This project demonstrates that meaningful image descriptions can be generated through zero-shot prompting techniques.
2. ✅ Yes! MedCLIP shows significantly better understanding of medical imaging compared to general CLIP models, proving the value of domain-specialized embeddings.
