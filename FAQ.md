# Frequently Asked Questions (FAQ)

## General Questions

### What is EJB-VLM?

EJB-VLM is a zero-shot vision-language model that combines pre-trained CLIP and GPT-2 to generate image descriptions without any training. It's a research project exploring whether meaningful VLMs can be built purely from existing components.

### Do I need to train the model?

No! That's the whole point. The model uses pre-trained CLIP and GPT-2 without any fine-tuning or training.

### What makes this different from other VLMs?

- **No training required**: Uses only pre-trained models
- **Fully parametrized**: All settings in YAML config
- **Multiple variants**: Standard and specialized (medical) versions
- **Modular design**: Easy to extend and customize

### Can I use this commercially?

Yes, under the MIT License. However, check the licenses of the underlying models (CLIP, GPT-2, MedCLIP) for their specific terms.

## Installation & Setup

### What are the system requirements?

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB disk space

**Recommended:**
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- 10GB disk space

### Do I need a GPU?

No, but it's highly recommended. The model runs on CPU but will be significantly slower (5-10x).

### How much memory does it use?

- **CLIP ViT-B/32**: ~1.5GB GPU / 2GB RAM
- **CLIP ViT-L/14**: ~3GB GPU / 4GB RAM
- **MedCLIP**: ~2GB GPU / 3GB RAM
- **GPT-2**: ~0.5GB GPU / 1GB RAM

### Installation fails with CUDA errors?

Install CPU-only PyTorch first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Can I use a different Python version?

Python 3.8-3.11 are officially supported. Python 3.12+ may work but is untested.

## Usage Questions

### How do I get started quickly?

```bash
# Install
pip install -r requirements.txt

# Run demo
python examples/demo.py

# Try your image
python -c "
from src.models.ejb_vlm_model import EJBVLMDescriptor
model = EJBVLMDescriptor()
print(model.describe_image('your_image.jpg'))
"
```

### How do I improve description quality?

1. **Use detailed preset**: `preset="detailed"`
2. **Try larger models**: `clip_model_name="ViT-L/14"`, `gpt_model_name="gpt2-medium"`
3. **Adjust temperature**: Higher = more creative, Lower = more precise
4. **Increase beams**: `num_beams=10` for better quality (slower)

### Descriptions are too generic. How to fix?

```python
# More specific descriptions
model.describe_image("image.jpg", 
    preset="detailed",
    temperature=1.0,
    max_length=80
)
```

### How do I process multiple images efficiently?

```python
# Use batch processing
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = model.batch_describe_images(images)

# Or use the batch_processing example
python examples/batch_processing.py
```

### Can I customize the categories?

Yes! Edit `config/config.yaml`:

```yaml
categories:
  my_categories:
    - "my category 1"
    - "my category 2"
```

Then use:
```python
model = AdvancedEJBVLMDescriptor(category_set="my_categories")
```

## Medical Imaging

### What is MedCLIP?

MedCLIP is a CLIP variant trained on medical images (PubMed, MIMIC-CXR datasets). It understands medical terminology and anatomical structures better than general CLIP.

### Is it accurate for diagnosis?

**NO!** This tool is for research/education only. Never use it for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals.

### What medical images does it support?

- Chest X-rays
- CT scans
- MRI images
- Ultrasound
- Histopathology slides
- Microscopy images
- Pathology slides

### How do I use MedCLIP?

```python
medical_model = EJBVLMDescriptor(
    clip_model_name="flaviagiammarino/pubmed-clip-vit-base-patch32"
)
description = medical_model.describe_image("xray.jpg")
```

### Can I add more medical models?

Yes! Add them to `config/config.yaml` variants section and follow the same pattern.

## Performance

### Why is inference slow?

Common causes:
1. **Running on CPU**: Use GPU for 5-10x speedup
2. **Large beam search**: Reduce `num_beams`
3. **Large model**: Use smaller CLIP variant (RN50)
4. **High resolution**: Resize images before processing

### How to speed up processing?

```python
# Faster preset
model.describe_image("img.jpg", preset="creative")

# Reduce beams
model.describe_image("img.jpg", num_beams=3)

# Use smaller model
model = EJBVLMDescriptor(clip_model_name="RN50")
```

### Can I run multiple instances?

Yes, but each instance loads the full model into memory. Better to process batches with a single instance.

### Does it cache models?

Yes! First load downloads and caches models. Subsequent runs are much faster.

## Customization

### Can I use different CLIP models?

Yes! Available models:
- ResNet: RN50, RN101, RN50x4, RN50x16, RN50x64
- ViT: ViT-B/32, ViT-B/16, ViT-L/14
- Specialized: MedCLIP, others from Hugging Face

### Can I use a different language model?

Currently supports GPT-2 variants (gpt2, gpt2-medium, gpt2-large, gpt2-xl). Adding other models (LLaMA, Mistral) would require code modifications.

### How do I create custom variants?

Add to `config/config.yaml`:

```yaml
variants:
  my_variant:
    clip_model: "model-name"
    description: "My custom variant"
    category_set: "my_categories"
    template_set: "my_templates"
```

### Can I fine-tune the models?

The current implementation doesn't include fine-tuning. You could add it, but that defeats the "zero-shot" purpose of the project.

## Troubleshooting

### Out of memory error?

```python
# Use CPU
model = EJBVLMDescriptor(device="cpu")

# Or smaller model
model = EJBVLMDescriptor(clip_model_name="RN50")
```

### MedCLIP not loading?

Check internet connection (downloads from Hugging Face on first use). Or manually download:
```bash
git lfs install
git clone https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32
```

### Import errors?

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or specific package
pip install --upgrade transformers
```

### CLIP model not found?

Ensure model name is correct. Check available models:
```python
import clip
print(clip.available_models())
```

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. Ways to help:
- Report bugs
- Suggest features
- Add model variants
- Improve documentation
- Submit code improvements

### Can I add my own CLIP variant?

Absolutely! We welcome additions of specialized CLIP models (art, satellite, fashion, etc.).

### Where do I report bugs?

Open an issue on GitHub with:
- Description of the bug
- Steps to reproduce
- System information
- Error messages

## Advanced

### Can I access raw embeddings?

Yes:
```python
embedding = model.encode_image("image.jpg")
# Returns torch.Tensor of CLIP features
```

### How do I compare images programmatically?

```python
from src.models.ejb_vlm_model import AdvancedEJBVLMDescriptor

model = AdvancedEJBVLMDescriptor()
similarity = model.compare_images("img1.jpg", "img2.jpg")
# Returns float between 0-1
```

### Can I integrate this into my application?

Yes! It's designed to be modular. Import and use the descriptor classes in your code.

### Is there a REST API?

Not yet, but it's on the roadmap. You could easily wrap it with Flask/FastAPI.

### Can I export to ONNX?

Not officially supported yet, but CLIP can be exported. Check the roadmap for updates.

## Still Have Questions?

- Open an issue on GitHub
- Check the [README](README.md) for more documentation
- See [examples/](examples/) for more code samples
- Contact [@edujbarrIos](https://github.com/edujbarrIos)
