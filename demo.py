"""
Quick Demo Script for EJB Vision-Language Model
Downloads a sample image and generates descriptions

Author: Eduardo J. Barrios (@edujbarruos)
"""

from ejb_vlm_model import CLIPGPTDescriptor, AdvancedCLIPGPTDescriptor
from utils import print_device_info, download_sample_image
import os


def demo():
    """Run a quick demo of the model."""
    print("\n" + "=" * 70)
    print("EJB VISION-LANGUAGE MODEL - QUICK DEMO")
    print("Author: Eduardo J. Barrios (@edujbarruos)")
    print("=" * 70)
    
    # Print device info
    print_device_info()
    
    # Download a sample image if needed
    sample_image = "demo_image.jpg"
    if not os.path.exists(sample_image):
        print("Downloading sample image...")
        # Sample image from Unsplash (free to use)
        sample_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800"
        downloaded = download_sample_image(sample_url, sample_image)
        if not downloaded:
            print("\nCouldn't download sample image.")
            print("Please place an image named 'demo_image.jpg' in this directory.")
            print("Or run with your own image:\n")
            print("  python demo.py path/to/your/image.jpg")
            return
    else:
        print(f"Using existing image: {sample_image}")
    
    # Check if user provided their own image
    import sys
    if len(sys.argv) > 1:
        sample_image = sys.argv[1]
        print(f"Using user-provided image: {sample_image}")
    
    print("\n" + "-" * 70)
    print("INITIALIZING MODELS...")
    print("-" * 70)
    
    # Initialize advanced model
    model = AdvancedCLIPGPTDescriptor()
    
    print("\n" + "-" * 70)
    print("ANALYZING IMAGE...")
    print("-" * 70)
    
    # Get detailed description
    result = model.detailed_description(sample_image, num_descriptions=3)
    
    print(f"\nImage: {sample_image}")
    print("\n📊 DETECTED CATEGORIES:")
    for i, (category, confidence) in enumerate(result["categories"], 1):
        bar_length = int(confidence * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {i}. {category:15s} {bar} {confidence:.1%}")
    
    print("\n📝 GENERATED DESCRIPTIONS:")
    for i, desc in enumerate(result["descriptions"], 1):
        print(f"\n  {i}. {desc}")
    
    print("\n" + "-" * 70)
    print("COMPARING GENERATION PARAMETERS...")
    print("-" * 70)
    
    # Try different parameters
    print("\n🎯 With Higher Temperature (more creative):")
    creative_desc = model.describe_image(
        sample_image, 
        temperature=1.0, 
        max_length=60
    )
    print(f"  {creative_desc}")
    
    print("\n❄️  With Lower Temperature (more focused):")
    focused_desc = model.describe_image(
        sample_image, 
        temperature=0.5, 
        max_length=60
    )
    print(f"  {focused_desc}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nTo explore more features:")
    print("  python example_usage.py interactive")
    print("\nOr see example_usage.py for more examples:")
    print("  python example_usage.py advanced")
    print()


if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        print("\nMake sure you have installed all requirements:")
        print("  pip install -r requirements.txt")
