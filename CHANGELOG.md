# Changelog

All notable changes to EJB-VLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-11

### Added
- **MedCLIP Support**: Full integration of medical imaging variant
  - Medical-specific CLIP model (PubMed-trained)
  - Medical category set with radiological and pathological terms
  - Medical templates with clinical terminology
- **Model Variants System**: Parametrized configuration for different domains
  - Standard variant for general images
  - Medical variant for clinical images
- **Demo Scripts**: New example files
  - `demo_medical.py`: Medical imaging demonstrations
  - `demo_variants.py`: Variant switching examples
- **Documentation**: Comprehensive README updates
  - Medical use cases and disclaimer
  - Performance comparison tables
  - API reference documentation
  - Troubleshooting guide
  - Contributing guidelines
  - Citation information

### Changed
- Improved configuration structure with variants section
- Enhanced category detection with domain-specific options
- Updated examples with medical imaging workflows

### Documentation
- Added project badges
- Added roadmap and future enhancements
- Added performance benchmarks
- Added related work citations

## [1.0.0] - 2026-03-10

### Added
- Initial release of EJB-VLM
- CLIP + GPT-2 zero-shot image description
- Parametrized configuration via YAML
- Multiple generation presets (default, creative, focused, detailed)
- Category detection using CLIP's zero-shot classification
- Image similarity comparison
- Batch processing capabilities
- Advanced descriptor with detailed analysis
- Modular architecture with utility modules
- Configuration loader for centralized parameters
- Device detection and automatic GPU/CPU selection
- Image preprocessing utilities
- Demo scripts and examples
- Comprehensive documentation

### Core Components
- `EJBVLMDescriptor`: Basic VLM implementation
- `AdvancedEJBVLMDescriptor`: Extended VLM with category detection
- Configuration system with YAML support
- Utility modules for images, devices, config, and I/O

### Examples
- `demo.py`: Quick demonstration script
- `example_usage.py`: Comprehensive usage examples
- Interactive mode for testing

[1.1.0]: https://github.com/edujbarrIos/ejb-vlm/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/edujbarrIos/ejb-vlm/releases/tag/v1.0.0
