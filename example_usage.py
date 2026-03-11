"""
Example usage of the EJB Vision-Language Model
Author: Eduardo J. Barrios (@edujbarruos)
"""

from ejb_vlm_model import CLIPGPTDescriptor, AdvancedCLIPGPTDescriptor
import sys


def basic_example():
    """Basic example: Generate a single description for an image."""
    print("=" * 60)
    print("BASIC EXAMPLE: Single Image Description")
    print("=" * 60)
    
    # Initialize the model
    model = CLIPGPTDescriptor()
    
    # Example image path (replace with your own)
    image_path = "example_image.jpg"
    
    print(f"\nGenerating description for: {image_path}")
    try:
        description = model.describe_image(image_path)
        print(f"\nDescription: {description}\n")
    except FileNotFoundError:
        print(f"\nImage not found: {image_path}")
        print("Please provide a valid image path.\n")


def multiple_descriptions_example():
    """Generate multiple alternative descriptions."""
    print("=" * 60)
    print("MULTIPLE DESCRIPTIONS EXAMPLE")
    print("=" * 60)
    
    model = CLIPGPTDescriptor()
    image_path = "example_image.jpg"
    
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
        print("Please provide a valid image path.\n")


def advanced_example():
    """Advanced example with category detection."""
    print("=" * 60)
    print("ADVANCED EXAMPLE: Category Detection + Description")
    print("=" * 60)
    
    # Initialize advanced model
    model = AdvancedCLIPGPTDescriptor()
    
    image_path = "example_image.jpg"
    
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
        print("Please provide a valid image path.\n")


def batch_processing_example():
    """Process multiple images at once."""
    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    model = CLIPGPTDescriptor()
    
    # Example image paths (replace with your own)
    image_paths = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg"
    ]
    
    print(f"\nProcessing {len(image_paths)} images...")
    results = model.batch_describe_images(image_paths, max_length=50)
    
    print("\nResults:")
    for path, description in results.items():
        print(f"\n{path}:")
        print(f"  {description}")
    print()


def compare_images_example():
    """Compare similarity between two images."""
    print("=" * 60)
    print("IMAGE COMPARISON EXAMPLE")
    print("=" * 60)
    
    model = CLIPGPTDescriptor()
    
    image1 = "image1.jpg"
    image2 = "image2.jpg"
    
    print(f"\nComparing:")
    print(f"  Image 1: {image1}")
    print(f"  Image 2: {image2}")
    
    try:
        similarity = model.compare_images(image1, image2)
        print(f"\nCosine Similarity: {similarity:.4f}")
        print(f"Similarity Percentage: {similarity * 100:.2f}%\n")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please provide valid image paths.\n")


def interactive_mode():
    """Interactive mode to describe any image."""
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("\nInitializing model...")
    
    model = AdvancedCLIPGPTDescriptor()
    
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
    print("EJB Vision-Language Model - Example Usage")
    print("Author: Eduardo J. Barrios (@edujbarruos)")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        examples = {
            "basic": basic_example,
            "multiple": multiple_descriptions_example,
            "advanced": advanced_example,
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
    print("\nUsage: python example_usage.py [mode]")
    print("\nAvailable modes:")
    print("  basic       - Basic single image description")
    print("  multiple    - Generate multiple descriptions")
    print("  advanced    - Advanced with category detection")
    print("  batch       - Process multiple images")
    print("  compare     - Compare two images")
    print("  interactive - Interactive mode")
    print("\nExample:")
    print("  python example_usage.py interactive")
    print()


if __name__ == "__main__":
    main()
