"""
Medical Image Analysis Demo using MedCLIP
Demonstrates the use of EJB-VLM with MedCLIP for analyzing medical images.

Author: Eduardo J. Barrios (@edujbarrIos)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ejb_vlm_model import EJBVLMDescriptor, AdvancedEJBVLMDescriptor
from src.utils.config_loader import load_config


def demo_basic_medical():
    """Basic medical image analysis with MedCLIP."""
    print("=" * 60)
    print("Medical Image Analysis with MedCLIP - Basic Demo")
    print("=" * 60)
    
    # Initialize with MedCLIP
    print("\n🏥 Initializing MedCLIP-based model...")
    model = EJBVLMDescriptor(
        clip_model_name="flaviagiammarino/pubmed-clip-vit-base-patch32"
    )
    
    print("\n📝 Note: This model is optimized for medical images such as:")
    print("   - Chest X-rays")
    print("   - CT scans")
    print("   - MRI images")
    print("   - Histopathology slides")
    print("   - Microscopy images")
    print("   - Ultrasound images")
    
    print("\n💡 To test with your own medical images:")
    print("   1. Place medical images in the 'data/' folder")
    print("   2. Update the image_path variable below")
    print("   3. Run this script")
    
    # Note: Users should provide their own medical images
    print("\n⚠️  Please provide your own medical image path to test.")
    print("   Example: model.describe_image('data/chest_xray.jpg')")
    

def demo_advanced_medical():
    """Advanced medical image analysis with category detection."""
    print("\n" + "=" * 60)
    print("Advanced Medical Image Analysis")
    print("=" * 60)
    
    # Initialize advanced model with medical categories
    print("\n🔬 Initializing advanced MedCLIP model with medical categories...")
    model = AdvancedEJBVLMDescriptor(
        clip_model_name="flaviagiammarino/pubmed-clip-vit-base-patch32",
        category_set="medical"
    )
    
    print("\n📊 Medical categories available:")
    config = load_config()
    medical_categories = config["categories"]["medical"]
    for i, category in enumerate(medical_categories, 1):
        print(f"   {i}. {category}")
    
    print("\n💡 Example usage:")
    print("   result = model.detailed_description('xray.jpg', num_descriptions=3)")
    print("   print('Modality:', result['categories'])")
    print("   print('Findings:', result['descriptions'])")
    

def demo_comparison():
    """Compare standard CLIP vs MedCLIP on medical images."""
    print("\n" + "=" * 60)
    print("Comparison: Standard CLIP vs MedCLIP")
    print("=" * 60)
    
    print("\n📊 Performance differences:")
    print("\n   Standard CLIP (ViT-B/32):")
    print("   ✓ Trained on general images (LAION, OpenAI datasets)")
    print("   ✓ Best for: everyday photos, scenes, objects")
    print("   ✗ Limited medical terminology understanding")
    print("   ✗ May misclassify medical imaging modalities")
    
    print("\n   MedCLIP (PubMed-trained):")
    print("   ✓ Trained on medical images (PubMed, MIMIC-CXR)")
    print("   ✓ Best for: radiological images, pathology, microscopy")
    print("   ✓ Understands medical terminology and anatomy")
    print("   ✓ Better zero-shot classification of medical conditions")
    print("   ✗ May underperform on general/everyday images")
    
    print("\n💡 Use case recommendation:")
    print("   - Use Standard CLIP for: photos, artwork, general objects")
    print("   - Use MedCLIP for: X-rays, CT/MRI scans, pathology slides")


def demo_medical_presets():
    """Demonstrate different generation presets for medical descriptions."""
    print("\n" + "=" * 60)
    print("Medical Description Generation Presets")
    print("=" * 60)
    
    print("\n📝 Available presets for medical image description:")
    
    print("\n   1. default: Balanced clinical descriptions")
    print("      - Temperature: 0.7, Beams: 5")
    print("      - Use for: General medical image analysis")
    
    print("\n   2. focused: Precise diagnostic-style descriptions")
    print("      - Temperature: 0.5, Beams: 8")
    print("      - Use for: Formal clinical reports")
    
    print("\n   3. detailed: Comprehensive anatomical descriptions")
    print("      - Max length: 80 tokens, Beams: 10")
    print("      - Use for: Educational or research purposes")
    
    print("\n   4. creative: Exploratory descriptions")
    print("      - Temperature: 1.0")
    print("      - Use for: When standard output is too conservative")
    
    print("\n💡 Example usage:")
    print("   model = EJBVLMDescriptor(clip_model_name='flaviagiammarino/pubmed-clip-vit-base-patch32')")
    print("   desc = model.describe_image('scan.jpg', preset='focused')")


def main():
    """Run all medical imaging demos."""
    print("\n" + "=" * 80)
    print(" " * 20 + "EJB-VLM Medical Imaging Demo")
    print(" " * 15 + "Using MedCLIP for Clinical Image Analysis")
    print("=" * 80)
    
    print("\n🎯 This demo showcases the medical imaging capabilities of EJB-VLM")
    print("   Author: Eduardo J. Barrios (@edujbarrIos)")
    
    try:
        # Run demo sections
        demo_basic_medical()
        demo_advanced_medical()
        demo_comparison()
        demo_medical_presets()
        
        print("\n" + "=" * 80)
        print("Demo Complete!")
        print("=" * 80)
        
        print("\n📚 Next steps:")
        print("   1. Add your medical images to the 'data/' folder")
        print("   2. Modify this script to load your images")
        print("   3. Experiment with different presets and parameters")
        print("   4. Check 'config/config.yaml' for customization options")
        
        print("\n⚠️  Medical Disclaimer:")
        print("   This tool is for research and educational purposes only.")
        print("   Not intended for clinical diagnosis or medical decision-making.")
        print("   Always consult qualified healthcare professionals for medical advice.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("   1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check that you have sufficient GPU memory (or use CPU)")
        print("   3. Verify MedCLIP model download: huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32")


if __name__ == "__main__":
    main()
