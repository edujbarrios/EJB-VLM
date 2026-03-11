# Output Directory

This directory stores processing results and outputs.

## Contents

- `descriptions.json` - Batch processing results
- `results_detailed.json` - Detailed analysis with categories
- `preset_comparison.json` - Preset comparison results
- `embeddings/` - CLIP embeddings if saved
- `reports/` - Generated reports

## Structure

```
output/
├── descriptions/
│   ├── batch_2026_03_11.json
│   └── ...
├── comparisons/
│   ├── similarity_matrix.json
│   └── ...
└── reports/
    ├── analysis_summary.md
    └── ...
```

## Note

This folder is ignored by git. Results are stored locally only.
