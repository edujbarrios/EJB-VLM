"""
Quick Demo Script for EJB Vision-Language Model
Downloads a sample image and generates descriptions

Author: Eduardo J. Barrios (@edujbarruos)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.ejb_vlm_model import CLIPGPTDescriptor, AdvancedCLIPGPTDescriptor
from src.utils.device_utils import print_device_info
from src.utils.image_utils import download_sample_image
from src.utils.config_loader import get_config


def demo():
    """Run a quick demo of the model."""
    print("\n" + "=" * 70)
    print("EJB-CLIPVISION - QUICK DEMO")
    print("Zero-Shot Vision-Language Model")
    print("Author: Eduardo J. Barrios (@edujbarruos)")
    print("=" * 70)
    
    # Print device info
    print_device_info()
    
    # Load config
    config = get_config()
    demo_config = config["demo"]
    
    # Download a sample image if needed
    sample_image = demo_config["default_image_name"]
    data_dir = config["paths"]["data_dir"]
    sample_path = os.path.join(data_dir, sample_image)
    
    if not os.path.exists(sample_path):
        print("Downloading sample image...")
        os.makedirs(data_dir, exist_ok=True)
        downloaded = download_sample_image(demo_config["sample_image_url"], sample_path)
        if not downloaded:
            print("\nCouldn't download sample image.")
            print(f"Please place an image named '{sample_image}' in the {data_dir}/ directory.")
            print("Or run with your own image:\n")
            print("  python examples/demo.py path/to/your/image.jpg")
            return
    else:
        print(f"Using existing image: {sample_path}")
    
    # Check if user provided their own image
    if len(sys.argv) > 1:
        sample_path = sys.argv[1]
        print(f"Using user-provided image: {sample_path}")
    
    print("\n" + "-" * 70)
    print("INITIALIZING MODELS...")
    print("-" * 70)
    
    # Initialize advanced model
    model = AdvancedCLIPGPTDescriptor()
    
    print("\n" + "-" * 70)
    print("ANALYZING IMAGE...")
    print("-" * 70)
    
    # Get detailed description
    result = model.detailed_description(
        sample_path, 
        num_descriptions=demo_config["num_descriptions"]
    )
    
    print(f"\nImage: {sample_path}")
    print("\n📊 DETECTED CATEGORIES:")
    for i, (category, confidence) in enumerate(result["categories"], 1):
        bar_length = int(confidence * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {i}. {category:15s} {bar} {confidence:.1%}")
    
    print("\n📝 GENERATED DESCRIPTIONS:")
    for i, desc in enumerate(result["descriptions"], 1):
        print(f"\n  {i}. {desc}")
    
    print("\n" + "-" * 70)
    print("COMPARING GENERATION PRESETS...")
    print("-" * 70)
    
    # Try different presets from config
    print("\n🎯 Creative Preset (more creative):")
    creative_desc = model.describe_image(sample_path, preset="creative")
    print(f"  {creative_desc}")
    
    print("\n❄️  Focused Preset (more focused):")
    focused_desc = model.describe_image(sample_path, preset="focused")
    print(f"  {focused_desc}")
    
    print("\n📖 Detailed Preset (longer description):")
    detailed_desc = model.describe_image(sample_path, preset="detailed")
    print(f"  {detailed_desc}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nTo explore more features:")
    print("  python examples/example_usage.py interactive")
    print("\nOr see example_usage.py for more examples:")
    print("  python examples/example_usage.py advanced")
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
        import traceback
        traceback.print_exc()
