"""
Example: Using Model Variants from Configuration
Demonstrates how to easily switch between Standard and Medical variants.

Author: Eduardo J. Barrios (@edujbarrIos)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ejb_vlm_model import EJBVLMDescriptor, AdvancedEJBVLMDescriptor
from src.utils.config_loader import load_config


def demo_standard_variant():
    """Demonstrate the standard CLIP variant for general images."""
    print("\n" + "=" * 60)
    print("Standard Variant - General Purpose VLM")
    print("=" * 60)
    
    config = load_config()
    standard_config = config["variants"]["standard"]
    
    print(f"\nConfiguration:")
    print(f"  Model: {standard_config['clip_model']}")
    print(f"  Description: {standard_config['description']}")
    print(f"  Categories: {standard_config['category_set']}")
    print(f"  Templates: {standard_config['template_set']}")
    
    # Initialize with standard CLIP
    print("\n✨ Initializing standard model...")
    model = EJBVLMDescriptor(
        clip_model_name=standard_config["clip_model"]
    )
    
    print("\n💡 Best for: everyday photos, artwork, general objects")
    print("   Example: model.describe_image('photo.jpg')")


def demo_medical_variant():
    """Demonstrate the medical MedCLIP variant for clinical images."""
    print("\n" + "=" * 60)
    print("Medical Variant - MedCLIP for Clinical Images")
    print("=" * 60)
    
    config = load_config()
    medical_config = config["variants"]["medical"]
    
    print(f"\nConfiguration:")
    print(f"  Model: {medical_config['clip_model']}")
    print(f"  Description: {medical_config['description']}")
    print(f"  Categories: {medical_config['category_set']}")
    print(f"  Templates: {medical_config['template_set']}")
    
    # Initialize with MedCLIP
    print("\n🏥 Initializing medical model...")
    model = EJBVLMDescriptor(
        clip_model_name=medical_config["clip_model"]
    )
    
    print("\n💡 Best for: X-rays, CT scans, MRI, pathology slides")
    print("   Example: model.describe_image('chest_xray.jpg')")


def demo_easy_switching():
    """Show how to easily switch between variants."""
    print("\n" + "=" * 60)
    print("Easy Variant Switching")
    print("=" * 60)
    
    print("\n📝 Method 1: Direct specification")
    print("   # Standard")
    print("   model = EJBVLMDescriptor(clip_model_name='ViT-B/32')")
    print("   ")
    print("   # Medical")
    print("   model = EJBVLMDescriptor(clip_model_name='flaviagiammarino/pubmed-clip-vit-base-patch32')")
    
    print("\n📝 Method 2: Load from config")
    print("   config = load_config()")
    print("   variant = config['variants']['medical']  # or 'standard'")
    print("   model = EJBVLMDescriptor(clip_model_name=variant['clip_model'])")
    
    print("\n📝 Method 3: Advanced with categories")
    print("   config = load_config()")
    print("   variant = config['variants']['medical']")
    print("   model = AdvancedEJBVLMDescriptor(")
    print("       clip_model_name=variant['clip_model'],")
    print("       category_set=variant['category_set']")
    print("   )")


def demo_custom_variant():
    """Show how users can define their own variants."""
    print("\n" + "=" * 60)
    print("Creating Custom Variants")
    print("=" * 60)
    
    print("\n💡 You can add your own variants in config/config.yaml:")
    print("\nExample - Add a 'lightweight' variant:")
    print("""
variants:
  lightweight:
    clip_model: "RN50"  # Smaller, faster model
    description: "Fast variant for quick processing"
    category_set: "default"
    template_set: "basic"
    gpt_model: "gpt2"  # Instead of larger variants
""")
    
    print("\nExample - Add a 'detailed_medical' variant:")
    print("""
variants:
  detailed_medical:
    clip_model: "flaviagiammarino/pubmed-clip-vit-base-patch32"
    description: "Detailed medical analysis with comprehensive output"
    category_set: "medical"
    template_set: "medical"
    gpt_model: "gpt2-medium"  # Larger model for better descriptions
""")


def main():
    """Run all variant demos."""
    print("\n" + "=" * 80)
    print(" " * 25 + "Model Variants Demo")
    print(" " * 20 + "Parametrized Configuration System")
    print("=" * 80)
    
    print("\n🎯 This demo shows how to use the parametrized variant system")
    print("   to easily switch between different model configurations.")
    
    try:
        demo_standard_variant()
        demo_medical_variant()
        demo_easy_switching()
        demo_custom_variant()
        
        print("\n" + "=" * 80)
        print("Demo Complete!")
        print("=" * 80)
        
        print("\n📚 Key Benefits of Parametrized Variants:")
        print("   ✓ No code changes needed to switch models")
        print("   ✓ Consistent configuration across projects")
        print("   ✓ Easy to add new specialized variants")
        print("   ✓ Domain-specific optimizations (medical, art, satellite, etc.)")
        
        print("\n🔧 Next Steps:")
        print("   1. Check config/config.yaml to see variant definitions")
        print("   2. Try both standard and medical variants on your images")
        print("   3. Create custom variants for your specific use case")
        print("   4. Experiment with different generation presets")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
