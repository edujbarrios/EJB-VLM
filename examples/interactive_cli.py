"""
Interactive CLI for EJB-VLM
Command-line interface for easy model interaction.

Author: Eduardo J. Barrios (@edujbarrIos)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ejb_vlm_model import EJBVLMDescriptor, AdvancedEJBVLMDescriptor
from src.utils.config_loader import load_config


class InteractiveCLI:
    """Interactive command-line interface for EJB-VLM."""
    
    def __init__(self):
        self.model = None
        self.advanced_model = None
        self.config = load_config()
        self.current_mode = "basic"
        
    def print_header(self):
        """Print welcome header."""
        print("\n" + "="*80)
        print(" "*25 + "EJB-VLM Interactive CLI")
        print(" "*20 + "Eduardo J. Barrios (@edujbarrIos)")
        print("="*80)
        
    def print_menu(self):
        """Print main menu."""
        print("\n📋 Available Commands:")
        print("  1. describe <image_path>        - Generate image description")
        print("  2. detailed <image_path>        - Detailed analysis with categories")
        print("  3. compare <img1> <img2>        - Compare two images")
        print("  4. batch <folder_path>          - Process all images in folder")
        print("  5. preset <name>                - Change generation preset")
        print("  6. model <basic|advanced|medical> - Switch model mode")
        print("  7. models                       - List available CLIP models")
        print("  8. config                       - Show current configuration")
        print("  9. help                         - Show this menu")
        print("  0. exit                         - Exit program")
        print()
        
    def initialize_model(self, mode="basic"):
        """Initialize the model based on mode."""
        print(f"\n🔧 Initializing {mode} model...")
        
        try:
            if mode == "basic":
                self.model = EJBVLMDescriptor()
                self.current_mode = "basic"
            elif mode == "advanced":
                self.model = AdvancedEJBVLMDescriptor(category_set="extended")
                self.current_mode = "advanced"
            elif mode == "medical":
                self.model = EJBVLMDescriptor(
                    clip_model_name="flaviagiammarino/pubmed-clip-vit-base-patch32"
                )
                self.current_mode = "medical"
            
            print(f"✅ {mode.capitalize()} model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        
        return True
    
    def describe_image(self, image_path, preset="default"):
        """Generate description for an image."""
        if not self.model:
            print("❌ Model not initialized. Use 'model basic' first.")
            return
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"\n🖼️  Analyzing: {image_path}")
        print(f"⚙️  Preset: {preset}")
        
        try:
            description = self.model.describe_image(image_path, preset=preset)
            print(f"\n📝 Description:\n   {description}\n")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def detailed_analysis(self, image_path):
        """Perform detailed analysis with categories."""
        if self.current_mode != "advanced":
            print("⚠️  Switching to advanced mode...")
            if not self.initialize_model("advanced"):
                return
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"\n🔍 Detailed analysis: {image_path}")
        
        try:
            result = self.model.detailed_description(
                image_path,
                num_descriptions=2,
                top_categories=5
            )
            
            print(f"\n🏷️  Categories:")
            for i, cat in enumerate(result["categories"], 1):
                print(f"   {i}. {cat}")
            
            print(f"\n📝 Descriptions:")
            for i, desc in enumerate(result["descriptions"], 1):
                print(f"   {i}. {desc}")
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def compare_images(self, img1, img2):
        """Compare two images."""
        if self.current_mode != "advanced":
            print("⚠️  Switching to advanced mode...")
            if not self.initialize_model("advanced"):
                return
        
        if not os.path.exists(img1):
            print(f"❌ Image not found: {img1}")
            return
        if not os.path.exists(img2):
            print(f"❌ Image not found: {img2}")
            return
        
        print(f"\n🔄 Comparing images...")
        print(f"   Image 1: {img1}")
        print(f"   Image 2: {img2}")
        
        try:
            similarity = self.model.compare_images(img1, img2)
            print(f"\n📊 Similarity Score: {similarity:.4f}")
            
            if similarity > 0.9:
                print("   → Very similar images!")
            elif similarity > 0.7:
                print("   → Similar images")
            elif similarity > 0.5:
                print("   → Somewhat similar")
            else:
                print("   → Different images")
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def batch_process(self, folder_path):
        """Process all images in a folder."""
        if not self.model:
            print("❌ Model not initialized. Use 'model basic' first.")
            return
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = [
            str(f) for f in Path(folder_path).rglob('*')
            if f.suffix.lower() in image_extensions
        ]
        
        if not images:
            print(f"⚠️  No images found in {folder_path}")
            return
        
        print(f"\n📊 Found {len(images)} images")
        confirm = input("Process all? (y/n): ")
        
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
        
        print("\n🔄 Processing...")
        results = self.model.batch_describe_images(images)
        
        print("\n✅ Results:")
        for path, desc in results.items():
            print(f"\n{Path(path).name}:")
            print(f"  {desc}")
        print()
    
    def list_models(self):
        """List available CLIP models."""
        print("\n🤖 Available CLIP Models:")
        print("\nGeneral Purpose:")
        print("  - RN50, RN101, RN50x4, RN50x16, RN50x64")
        print("  - ViT-B/32, ViT-B/16, ViT-L/14")
        print("\nSpecialized:")
        print("  - flaviagiammarino/pubmed-clip-vit-base-patch32 (Medical)")
        print("\n💡 Configure in config/config.yaml or pass to model init")
        print()
    
    def show_config(self):
        """Display current configuration."""
        print("\n⚙️  Current Configuration:")
        print(f"\n  Mode: {self.current_mode}")
        
        if self.model:
            print(f"  CLIP Model: {self.model.clip_model_name}")
            print(f"  GPT Model: {self.model.gpt_model_name}")
            print(f"  Device: {self.model.device}")
        
        print(f"\n📁 Paths:")
        print(f"  Data: {self.config['paths']['data_dir']}")
        print(f"  Output: {self.config['paths']['output_dir']}")
        print()
    
    def run(self):
        """Run the interactive CLI."""
        self.print_header()
        
        print("\n👋 Welcome to EJB-VLM Interactive Mode!")
        print("   Type 'help' to see available commands")
        
        # Initialize basic model
        if not self.initialize_model("basic"):
            return
        
        current_preset = "default"
        
        while True:
            try:
                command = input("\nejb-vlm> ").strip()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0].lower()
                
                if cmd == "exit" or cmd == "0":
                    print("\n👋 Goodbye!")
                    break
                
                elif cmd == "help" or cmd == "9":
                    self.print_menu()
                
                elif cmd == "describe" or cmd == "1":
                    if len(parts) < 2:
                        print("Usage: describe <image_path>")
                    else:
                        self.describe_image(parts[1], current_preset)
                
                elif cmd == "detailed" or cmd == "2":
                    if len(parts) < 2:
                        print("Usage: detailed <image_path>")
                    else:
                        self.detailed_analysis(parts[1])
                
                elif cmd == "compare" or cmd == "3":
                    if len(parts) < 3:
                        print("Usage: compare <image1> <image2>")
                    else:
                        self.compare_images(parts[1], parts[2])
                
                elif cmd == "batch" or cmd == "4":
                    if len(parts) < 2:
                        print("Usage: batch <folder_path>")
                    else:
                        self.batch_process(parts[1])
                
                elif cmd == "preset" or cmd == "5":
                    if len(parts) < 2:
                        print(f"Current preset: {current_preset}")
                        print("Available: default, creative, focused, detailed")
                    else:
                        current_preset = parts[1]
                        print(f"✅ Preset changed to: {current_preset}")
                
                elif cmd == "model" or cmd == "6":
                    if len(parts) < 2:
                        print(f"Current mode: {self.current_mode}")
                        print("Available: basic, advanced, medical")
                    else:
                        self.initialize_model(parts[1])
                
                elif cmd == "models" or cmd == "7":
                    self.list_models()
                
                elif cmd == "config" or cmd == "8":
                    self.show_config()
                
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("   Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Use 'exit' to quit.")
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Entry point for CLI."""
    cli = InteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
