# Data Directory

This directory is for storing images and datasets.

## Usage

Place your images here to test the model:

```
data/
├── sample_images/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
├── medical_images/
│   ├── xray1.jpg
│   ├── ct_scan1.jpg
│   └── ...
└── test_images/
    └── ...
```

## Important Notes

- This folder is ignored by git (see .gitignore)
- Do not commit sensitive or confidential images
- Organize images into subfolders for easier management
- Supported formats: .jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp

## Sample Images

For testing purposes, you can:
1. Download from Unsplash, Pexels (free stock photos)
2. Use the demo script: `python examples/demo.py` (auto-downloads sample)
3. Use your own images (ensure you have rights to use them)

## Medical Images

If testing medical imaging:
- Use publicly available datasets (MIMIC-CXR, PadChest, etc.)
- Ensure proper data use agreements
- Never commit PHI/PII data
- Follow HIPAA/GDPR guidelines if applicable

## Clean Up

To free disk space:
```bash
# Remove all images (Windows)
del /s data\*.jpg data\*.png data\*.bmp

# Remove all images (Linux/Mac)
find data/ -type f \( -name "*.jpg" -o -name "*.png" \) -delete
```
