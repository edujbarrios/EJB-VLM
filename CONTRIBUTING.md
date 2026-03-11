# Contributing to EJB-VLM

First off, thank you for considering contributing to EJB-VLM! It's people like you that make this project better for everyone.

## Code of Conduct

This project and everyone participating in it is governed by respect, collaboration, and inclusivity. By participating, you are expected to uphold this standard.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load model with '...'
2. Process image '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 2.0.1]
- CUDA version (if applicable): [e.g., 11.8]

**Additional context**
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives**: What other solutions have you considered?
- **Impact**: Who benefits from this enhancement?

### Adding New Model Variants

We welcome contributions of new CLIP variants for specialized domains! To add a new variant:

1. **Research**: Find a suitable CLIP variant on Hugging Face or elsewhere
2. **Configuration**: Add it to `config/config.yaml`:

```yaml
variants:
  your_domain:
    clip_model: "organization/model-name"
    description: "Brief description of the domain"
    category_set: "your_categories"
    template_set: "your_templates"
```

3. **Categories**: Add domain-specific categories:

```yaml
categories:
  your_categories:
    - "category 1"
    - "category 2"
    # ...
```

4. **Templates**: Add domain-specific templates:

```yaml
templates:
  your_templates:
    - "template 1"
    - "template 2"
    # ...
```

5. **Demo**: Create a demo file in `examples/demo_your_domain.py`
6. **Documentation**: Update README.md with your variant information
7. **Testing**: Test thoroughly with domain-specific images

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow the code style guidelines
   - Add/update tests if applicable
   - Update documentation
4. **Commit** with clear messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```
5. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** with:
   - Clear title and description
   - Reference any related issues
   - Screenshot/example if applicable

## Development Guidelines

### Code Style

- Follow **PEP 8** style guide
- Use **type hints** where possible
- Write **docstrings** for all public functions/classes
- Keep functions focused and modular
- Use meaningful variable names

**Example:**

```python
def describe_image(
    self, 
    image_path: str, 
    preset: Optional[str] = None, 
    **kwargs
) -> Union[str, List[str]]:
    """
    Generate a description for an image.
    
    Args:
        image_path: Path to the image file
        preset: Generation preset name (default, creative, focused, detailed)
        **kwargs: Additional generation parameters
        
    Returns:
        Generated description(s) as string or list of strings
        
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If preset is not recognized
    """
    # Implementation
```

### Configuration

- **Never hardcode** values - use `config.yaml`
- Keep configuration **hierarchical** and **logical**
- Add **comments** explaining configuration options
- Provide **sensible defaults**

### Testing

Currently, the project is setting up its testing infrastructure. When contributing:

- Manually test your changes with various inputs
- Verify GPU and CPU compatibility
- Test edge cases and error handling
- Document your testing process in the PR

### Documentation

- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Keep CHANGELOG.md updated

## Project Structure

```
ejb-vlm/
├── config/           # Configuration files
├── src/              # Source code
│   ├── models/       # Model implementations
│   └── utils/        # Utility modules
├── examples/         # Example scripts and demos
├── data/             # Data directory (not committed)
├── tests/            # Test files
└── docs/             # Additional documentation
```

### Adding New Modules

When adding a new module:

1. Place in appropriate directory (`models/`, `utils/`, etc.)
2. Add `__init__.py` imports
3. Follow existing naming conventions
4. Include comprehensive docstrings
5. Add usage examples

## Recognition

Contributors will be acknowledged in:
- README.md (if significant contribution)
- CHANGELOG.md (for all contributions)
- Project documentation

## Questions?

Don't hesitate to ask! You can:
- Open an issue with the `question` label
- Reach out to [@edujbarrIos](https://github.com/edujbarrIos)

Thank you for contributing! 🎉
