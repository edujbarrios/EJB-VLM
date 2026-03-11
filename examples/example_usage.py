"""
Example usage of the EJB Vision-Language Model
Author: Eduardo J. Barrios (@edujbarrIos)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.ejb_vlm_model import EJBVLMDescriptor, AdvancedEJBVLMDescriptor
from src.utils.config_loader import get_config


def basic_example():
    """Basic example: Generate a single description for an image."""
    print("=" * 60)
    print("BASIC EXAMPLE: Single Image Description")
    print("=" * 60)
    
    # Initialize the model
    model = EJBVLMDescriptor()
    
    # Example image path (replace with your own)
    config = get_config()
    data_dir = config["paths"]["data_dir"]
    image_path = os.path.join(data_dir, "demo_image.jpg")
    
    print(f"\nGenerating description for: {image_path}")
    try:
        description = model.describe_image(image_path)
        print(f"\nDescription: {description}\n")
    except FileNotFoundError:
        print(f"\nImage not found: {image_path}")
        print("Please run the demo first: python examples/demo.py")
        print("Or provide a valid image path.\n")


def multiple_descriptions_example():
    """Generate multiple alternative descriptions."""
    print("=" * 60)
    print("MULTIPLE DESCRIPTIONS EXAMPLE")
    print("=" * 60)
    
    model = EJBVLMDescriptor()
    
    config = get_config()
    data_dir = config["paths"]["data_dir"]
    image_path = os.path.join(data_dir, "demo_image.jpg")
    
    print(f"\nGenerating 3 descriptions for: {image_path}")
    try:
        descriptions = model.describe_image(
            image_path, 
            num_return_sequences=3,
            max_length=60
        )
        
        for i, desc in enumerate(descriptions, 1):
            print(f"\n{i}. {desc}")
        print()
    except FileNotFoundError:
        print(f"\nImage not found: {image_path}")
        print("Please run the demo first: python examples/demo.py\n")


def advanced_example():
    """Advanced example with category detection."""
    print("=" * 60)
    print("ADVANCED EXAMPLE: Category Detection + Description")
    print("=" * 60)
    
    # Initialize advanced model with extended categories
    model = AdvancedEJBVLMDescriptor(category_set="extended")
    
    config = get_config()
    data_dir = config["paths"]["data_dir"]
    image_path = os.path.join(data_dir, "demo_image.jpg")
    
    print(f"\nAnalyzing: {image_path}")
    try:
        result = model.detailed_description(image_path, num_descriptions=2)
        
        print("\nDetected Categories:")
        for category, confidence in result["categories"]:
            print(f"  - {category}: {confidence:.2%}")
        
        print("\nGenerated Descriptions:")
        for i, desc in enumerate(result["descriptions"], 1):
            print(f"  {i}. {desc}")
        print()
    except FileNotFoundError:
        print(f"\nImage not found: {image_path}")
        print("Please run the demo first: python examples/demo.py\n")


def preset_comparison_example():
    """Compare different generation presets."""
    print("=" * 60)
    print("PRESET COMPARISON EXAMPLE")
    print("=" * 60)
    
    model = EJBVLMDescriptor()
    
    config = get_config()
    data_dir = config["paths"]["data_dir"]
    image_path = os.path.join(data_dir, "demo_image.jpg")
    
    presets = ["default", "creative", "focused", "detailed"]
    
    print(f"\nComparing presets for: {image_path}\n")
    
    try:
        for preset in presets:
            print(f"{preset.upper()}:")
            desc = model.describe_image(image_path, preset=preset)
            print(f"  {desc}\n")
    except FileNotFoundError:
        print(f"\nImage not found: {image_path}")
        print("Please run the demo first: python examples/demo.py\n")


def batch_processing_example():
    """Process multiple images at once."""
    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    model = EJBVLMDescriptor()
    
    config = get_config()
    data_dir = config["paths"]["data_dir"]
    
    # Load all images from data directory
    from src.utils.image_utils import load_images_from_directory
    
    try:
        image_paths = load_images_from_directory(data_dir)
        
        if not image_paths:
            print(f"\nNo images found in {data_dir}/")
            print("Please add some images or run the demo first.")
            return
        
        print(f"\nProcessing {len(image_paths)} images...")
        results = model.batch_describe_images(image_paths, max_length=50)
        
        print("\nResults:")
        for path, description in results.items():
            print(f"\n{os.path.basename(path)}:")
            print(f"  {description}")
        print()
    except Exception as e:
        print(f"\nError: {e}\n")


def compare_images_example():
    """Compare similarity between two images."""
    print("=" * 60)
    print("IMAGE COMPARISON EXAMPLE")
    print("=" * 60)
    
    model = EJBVLMDescriptor()
    
    config = get_config()
    data_dir = config["paths"]["data_dir"]
    
    # Need at least 2 images
    from src.utils.image_utils import load_images_from_directory
    
    try:
        image_paths = load_images_from_directory(data_dir)
        
        if len(image_paths) < 2:
            print(f"\nNeed at least 2 images in {data_dir}/ for comparison.")
            print("Please add more images.")
            return
        
        image1 = image_paths[0]
        image2 = image_paths[1]
        
        print(f"\nComparing:")
        print(f"  Image 1: {os.path.basename(image1)}")
        print(f"  Image 2: {os.path.basename(image2)}")
        
        similarity = model.compare_images(image1, image2)
        print(f"\nCosine Similarity: {similarity:.4f}")
        print(f"Similarity Percentage: {similarity * 100:.2f}%\n")
    except Exception as e:
        print(f"\nError: {e}\n")


def interactive_mode():
    """Interactive mode to describe any image."""
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("\nInitializing model...")
    
    model = AdvancedEJBVLMDescriptor()
    
    print("\nModel ready! Enter image paths to generate descriptions.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        image_path = input("Image path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not image_path:
            continue
        
        try:
            print("\nAnalyzing...")
            result = model.detailed_description(image_path, num_descriptions=1)
            
            print("\nCategories:", ", ".join([f"{cat} ({conf:.1%})" 
                                               for cat, conf in result["categories"]]))
            print("Description:", result["descriptions"][0])
            print()
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main function to run examples."""
    print("\n" + "=" * 60)
    print("EJB-VLM - Example Usage")
    print("Author: Eduardo J. Barrios (@edujbarrIos)")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        examples = {
            "basic": basic_example,
            "multiple": multiple_descriptions_example,
            "advanced": advanced_example,
            "presets": preset_comparison_example,
            "batch": batch_processing_example,
            "compare": compare_images_example,
            "interactive": interactive_mode
        }
        
        if mode in examples:
            examples[mode]()
        else:
            print(f"\nUnknown mode: {mode}")
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage instructions."""
    print("\nUsage: python examples/example_usage.py [mode]")
    print("\nAvailable modes:")
    print("  basic       - Basic single image description")
    print("  multiple    - Generate multiple descriptions")
    print("  advanced    - Advanced with category detection")
    print("  presets     - Compare different generation presets")
    print("  batch       - Process multiple images")
    print("  compare     - Compare two images")
    print("  interactive - Interactive mode")
    print("\nExample:")
    print("  python examples/example_usage.py interactive")
    print()


if __name__ == "__main__":
    main()
