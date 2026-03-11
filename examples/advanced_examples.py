"""
Advanced Example: Image Similarity and Clustering
Demonstrates how to use CLIP embeddings for image analysis.

Author: Eduardo J. Barrios (@edujbarrIos)
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ejb_vlm_model import EJBVLMDescriptor, AdvancedEJBVLMDescriptor


def find_similar_images(image_folder, reference_image, model, top_k=5):
    """
    Find images most similar to a reference image.
    
    Args:
        image_folder: Directory containing images
        reference_image: Path to reference image
        model: EJBVLMDescriptor instance
        top_k: Number of similar images to return
    """
    print(f"\n{'='*60}")
    print("Image Similarity Search")
    print(f"{'='*60}")
    
    # Get reference embedding
    print(f"\n🔍 Analyzing reference image: {reference_image}")
    ref_embedding = model.encode_image(reference_image)
    
    # Get all images in folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    images = [
        str(f) for f in Path(image_folder).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"⚠️  No images found in {image_folder}")
        return []
    
    print(f"📊 Comparing with {len(images)} images...\n")
    
    # Compute similarities
    similarities = []
    for img_path in images:
        if img_path == reference_image:
            continue
        try:
            img_embedding = model.encode_image(img_path)
            # Cosine similarity
            similarity = (ref_embedding @ img_embedding.T).item()
            similarities.append((img_path, similarity))
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    print(f"🏆 Top {top_k} most similar images:\n")
    for i, (img_path, score) in enumerate(similarities[:top_k], 1):
        print(f"{i}. {Path(img_path).name}")
        print(f"   Similarity: {score:.4f}")
        print(f"   Path: {img_path}\n")
    
    return similarities[:top_k]


def cluster_images(image_folder, model, num_clusters=3):
    """
    Cluster images based on visual similarity.
    
    Args:
        image_folder: Directory containing images
        model: EJBVLMDescriptor instance
        num_clusters: Number of clusters
    """
    print(f"\n{'='*60}")
    print("Image Clustering")
    print(f"{'='*60}")
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    images = [
        str(f) for f in Path(image_folder).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if len(images) < num_clusters:
        print(f"⚠️  Not enough images for {num_clusters} clusters")
        return
    
    print(f"\n📊 Clustering {len(images)} images into {num_clusters} groups...\n")
    
    # Get embeddings
    embeddings = []
    valid_images = []
    for img_path in images:
        try:
            embedding = model.encode_image(img_path)
            embeddings.append(embedding.cpu().numpy().flatten())
            valid_images.append(img_path)
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
    
    if len(embeddings) < num_clusters:
        print(f"⚠️  Not enough valid images for clustering")
        return
    
    embeddings = np.array(embeddings)
    
    # Simple K-means clustering (without sklearn dependency)
    print("🔄 Performing clustering...\n")
    
    # Initialize centroids randomly
    indices = np.random.choice(len(embeddings), num_clusters, replace=False)
    centroids = embeddings[indices]
    
    # K-means iterations
    for iteration in range(20):
        # Assign to clusters
        distances = np.array([
            np.linalg.norm(embeddings - centroid, axis=1)
            for centroid in centroids
        ])
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([
            embeddings[labels == i].mean(axis=0) if np.any(labels == i)
            else centroids[i]
            for i in range(num_clusters)
        ])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Display results
    print(f"✅ Clustering complete!\n")
    
    for cluster_id in range(num_clusters):
        cluster_images = [
            valid_images[i] for i in range(len(valid_images))
            if labels[i] == cluster_id
        ]
        print(f"📁 Cluster {cluster_id + 1} ({len(cluster_images)} images):")
        for img in cluster_images[:5]:  # Show first 5
            print(f"   - {Path(img).name}")
        if len(cluster_images) > 5:
            print(f"   ... and {len(cluster_images) - 5} more")
        print()


def analyze_image_collection(image_folder, model):
    """
    Comprehensive analysis of an image collection.
    
    Args:
        image_folder: Directory containing images
        model: AdvancedEJBVLMDescriptor instance
    """
    print(f"\n{'='*60}")
    print("Image Collection Analysis")
    print(f"{'='*60}")
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    images = [
        str(f) for f in Path(image_folder).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"⚠️  No images found in {image_folder}")
        return
    
    print(f"\n📊 Analyzing {len(images)} images...\n")
    
    # Analyze each image
    category_counts = {}
    all_descriptions = []
    
    for img_path in images[:10]:  # Limit to first 10 for demo
        print(f"🔍 Processing: {Path(img_path).name}")
        try:
            result = model.detailed_description(img_path, num_descriptions=1)
            
            # Count categories
            for category in result["categories"][:3]:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            all_descriptions.append({
                "path": img_path,
                "categories": result["categories"][:3],
                "description": result["descriptions"][0]
            })
            
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Collection Summary")
    print(f"{'='*60}\n")
    
    print("🏷️  Most Common Categories:")
    sorted_categories = sorted(
        category_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for category, count in sorted_categories[:5]:
        print(f"   {category}: {count} images")
    
    print(f"\n📝 Sample Descriptions:")
    for item in all_descriptions[:3]:
        print(f"\n   {Path(item['path']).name}")
        print(f"   Categories: {', '.join(item['categories'])}")
        print(f"   Description: {item['description']}")


def main():
    """Run advanced examples."""
    print("\n" + "="*80)
    print(" "*20 + "Advanced Image Analysis Examples")
    print("="*80)
    
    print("\n💡 This demo shows advanced usage patterns:")
    print("   1. Finding similar images")
    print("   2. Clustering images by visual similarity")
    print("   3. Analyzing image collections")
    
    print("\n⚠️  Note: These examples require a folder with multiple images.")
    print("   Place your images in the 'data/' folder to test.")
    
    # Example configuration
    image_folder = "data"
    
    # Check if folder exists and has images
    if not os.path.exists(image_folder):
        print(f"\n❌ Folder '{image_folder}' not found.")
        print("   Create the folder and add some images to test these features.")
        return
    
    # Initialize models
    print("\n🔧 Initializing models...")
    basic_model = EJBVLMDescriptor()
    advanced_model = AdvancedEJBVLMDescriptor(category_set="extended")
    
    print("\n" + "="*80)
    print("Examples Ready!")
    print("="*80)
    
    print("\n💡 Usage Examples:")
    
    print("\n1. Find similar images:")
    print("   from advanced_examples import find_similar_images")
    print("   similar = find_similar_images('data/', 'data/reference.jpg', model)")
    
    print("\n2. Cluster images:")
    print("   from advanced_examples import cluster_images")
    print("   cluster_images('data/', model, num_clusters=3)")
    
    print("\n3. Analyze collection:")
    print("   from advanced_examples import analyze_image_collection")
    print("   analyze_image_collection('data/', advanced_model)")
    
    # Uncomment to run automatically:
    # find_similar_images(image_folder, "reference.jpg", basic_model)
    # cluster_images(image_folder, basic_model)
    # analyze_image_collection(image_folder, advanced_model)


if __name__ == "__main__":
    main()
