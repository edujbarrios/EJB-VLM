"""
Batch Processing Example
Demonstrates efficient processing of multiple images.

Author: Eduardo J. Barrios (@edujbarrIos)
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ejb_vlm_model import EJBVLMDescriptor, AdvancedEJBVLMDescriptor


def batch_process_folder(
    folder_path,
    model,
    output_file="results.json",
    preset="default",
    save_embeddings=False
):
    """
    Process all images in a folder and save results.
    
    Args:
        folder_path: Path to folder containing images
        model: Model instance
        output_file: Path to save results
        preset: Generation preset
        save_embeddings: Whether to save CLIP embeddings
    """
    print(f"\n{'='*60}")
    print("Batch Image Processing")
    print(f"{'='*60}")
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    images = [
        str(f) for f in Path(folder_path).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"⚠️  No images found in {folder_path}")
        return
    
    print(f"\n📊 Found {len(images)} images to process")
    print(f"⚙️  Preset: {preset}")
    print(f"💾 Output: {output_file}\n")
    
    # Process images
    results = []
    start_time = time.time()
    
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing: {Path(img_path).name}")
        
        try:
            # Generate description
            description = model.describe_image(img_path, preset=preset)
            
            # Get embedding if requested
            embedding = None
            if save_embeddings:
                embedding_tensor = model.encode_image(img_path)
                embedding = embedding_tensor.cpu().numpy().tolist()
            
            result = {
                "filename": Path(img_path).name,
                "path": img_path,
                "description": description,
                "preset": preset,
                "timestamp": datetime.now().isoformat(),
                "embedding": embedding
            }
            
            results.append(result)
            print(f"   ✓ {description[:60]}...")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append({
                "filename": Path(img_path).name,
                "path": img_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Calculate stats
    elapsed = time.time() - start_time
    successful = len([r for r in results if "error" not in r])
    
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{len(images)}")
    print(f"⏱️  Time: {elapsed:.2f}s ({elapsed/len(images):.2f}s per image)")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")
    
    return results


def batch_process_with_categories(
    folder_path,
    model,
    output_file="results_detailed.json"
):
    """
    Process images with category detection.
    
    Args:
        folder_path: Path to folder containing images
        model: AdvancedEJBVLMDescriptor instance
        output_file: Path to save results
    """
    print(f"\n{'='*60}")
    print("Batch Processing with Category Detection")
    print(f"{'='*60}")
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    images = [
        str(f) for f in Path(folder_path).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"⚠️  No images found in {folder_path}")
        return
    
    print(f"\n📊 Found {len(images)} images to analyze\n")
    
    # Process images
    results = []
    start_time = time.time()
    
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Analyzing: {Path(img_path).name}")
        
        try:
            # Detailed analysis
            analysis = model.detailed_description(
                img_path,
                num_descriptions=2,
                top_categories=5
            )
            
            result = {
                "filename": Path(img_path).name,
                "path": img_path,
                "categories": analysis["categories"],
                "descriptions": analysis["descriptions"],
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            print(f"   Categories: {', '.join(analysis['categories'][:3])}")
            print(f"   Description: {analysis['descriptions'][0][:50]}...\n")
            
        except Exception as e:
            print(f"   ✗ Error: {e}\n")
            results.append({
                "filename": Path(img_path).name,
                "path": img_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Calculate stats
    elapsed = time.time() - start_time
    successful = len([r for r in results if "error" not in r])
    
    # Category statistics
    all_categories = []
    for result in results:
        if "categories" in result:
            all_categories.extend(result["categories"])
    
    category_counts = {}
    for cat in all_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{len(images)}")
    print(f"⏱️  Time: {elapsed:.2f}s\n")
    
    print("📊 Top Categories:")
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_cats[:5]:
        print(f"   {cat}: {count}")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "statistics": {
                "total_images": len(images),
                "successful": successful,
                "failed": len(images) - successful,
                "processing_time": elapsed,
                "category_counts": dict(sorted_cats)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


def compare_presets_batch(folder_path, model, sample_size=5):
    """
    Compare different presets on a sample of images.
    
    Args:
        folder_path: Path to folder containing images
        model: Model instance
        sample_size: Number of images to compare
    """
    print(f"\n{'='*60}")
    print("Preset Comparison on Multiple Images")
    print(f"{'='*60}")
    
    # Get images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    images = [
        str(f) for f in Path(folder_path).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"⚠️  No images found in {folder_path}")
        return
    
    # Sample images
    import random
    sample_images = random.sample(images, min(sample_size, len(images)))
    
    presets = ["default", "creative", "focused", "detailed"]
    
    print(f"\n📊 Comparing {len(presets)} presets on {len(sample_images)} images\n")
    
    results = {}
    
    for img_path in sample_images:
        filename = Path(img_path).name
        print(f"🖼️  {filename}")
        results[filename] = {}
        
        for preset in presets:
            try:
                desc = model.describe_image(img_path, preset=preset)
                results[filename][preset] = desc
                print(f"   {preset:10s}: {desc[:50]}...")
            except Exception as e:
                print(f"   {preset:10s}: Error - {e}")
        print()
    
    # Save comparison
    output_file = "results/preset_comparison.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Comparison saved to: {output_file}")
    
    return results


def main():
    """Run batch processing examples."""
    print("\n" + "="*80)
    print(" "*25 + "Batch Processing Demo")
    print("="*80)
    
    print("\n💡 This demo shows how to efficiently process multiple images:")
    print("   1. Basic batch processing")
    print("   2. Processing with category detection")
    print("   3. Comparing presets across images")
    
    # Check for images
    data_folder = "data"
    if not os.path.exists(data_folder):
        print(f"\n⚠️  Folder '{data_folder}' not found.")
        print("   Create the folder and add images to test batch processing.")
        return
    
    print("\n🔧 Initialize models and run examples:")
    
    print("\n1. Basic batch processing:")
    print("   model = EJBVLMDescriptor()")
    print("   batch_process_folder('data/', model, 'results/descriptions.json')")
    
    print("\n2. With category detection:")
    print("   advanced = AdvancedEJBVLMDescriptor(category_set='extended')")
    print("   batch_process_with_categories('data/', advanced)")
    
    print("\n3. Compare presets:")
    print("   compare_presets_batch('data/', model, sample_size=3)")


if __name__ == "__main__":
    main()
