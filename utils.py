"""
Utility functions for the EJB Vision-Language Model
Author: Eduardo J. Barrios (@edujbarruos)
"""

import os
from pathlib import Path
from PIL import Image
import torch
import json


def validate_image_path(image_path):
    """
    Validate that an image path exists and is a valid image file.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid image format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    ext = Path(image_path).suffix.lower()
    
    if ext not in valid_extensions:
        raise ValueError(f"Invalid image format: {ext}. Supported: {valid_extensions}")
    
    # Try to open the image to confirm it's valid
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}")


def get_device_info():
    """
    Get information about available compute devices.
    
    Returns:
        dict: Device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None
    }
    
    if info["cuda_available"]:
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
    
    return info


def print_device_info():
    """Print device information in a readable format."""
    info = get_device_info()
    print("\nDevice Information:")
    print(f"  CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"  Device Count: {info['device_count']}")
        print(f"  Current Device: {info['current_device']}")
        print(f"  Device Name: {info['device_name']}")
    else:
        print("  Using CPU")
    print()


def save_results_to_json(results, output_path="results.json"):
    """
    Save description results to a JSON file.
    
    Args:
        results (dict): Dictionary of results
        output_path (str): Path to output JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


def load_images_from_directory(directory, recursive=False):
    """
    Load all image paths from a directory.
    
    Args:
        directory (str): Path to directory
        recursive (bool): Whether to search recursively
        
    Returns:
        list: List of image paths
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_paths = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in valid_extensions:
                image_paths.append(file_path)
    
    return sorted(image_paths)


def resize_image(image_path, max_size=(800, 800), output_path=None):
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image_path (str): Path to input image
        max_size (tuple): Maximum (width, height)
        output_path (str): Path to save resized image. If None, overwrites original.
        
    Returns:
        str: Path to resized image
    """
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        if output_path is None:
            output_path = image_path
        
        img.save(output_path)
    
    return output_path


def create_image_grid(image_paths, grid_size=(2, 2), output_path="grid.jpg"):
    """
    Create a grid of images.
    
    Args:
        image_paths (list): List of image paths
        grid_size (tuple): (rows, cols) for grid
        output_path (str): Path to save grid
        
    Returns:
        str: Path to saved grid
    """
    from PIL import Image
    
    rows, cols = grid_size
    n_images = min(len(image_paths), rows * cols)
    
    # Load and resize images
    images = []
    target_size = (300, 300)
    
    for i in range(n_images):
        img = Image.open(image_paths[i]).convert('RGB')
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create a new image with exact target size (pad if needed)
        new_img = Image.new('RGB', target_size, (255, 255, 255))
        offset = ((target_size[0] - img.size[0]) // 2, 
                  (target_size[1] - img.size[1]) // 2)
        new_img.paste(img, offset)
        images.append(new_img)
    
    # Create grid
    grid_width = cols * target_size[0]
    grid_height = rows * target_size[1]
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * target_size[0]
        y = row * target_size[1]
        grid.paste(img, (x, y))
    
    grid.save(output_path)
    return output_path


def format_description(description, max_length=100):
    """
    Format a description to a maximum length.
    
    Args:
        description (str): Input description
        max_length (int): Maximum length
        
    Returns:
        str: Formatted description
    """
    if len(description) <= max_length:
        return description
    
    # Try to cut at a sentence boundary
    sentences = description.split('. ')
    result = sentences[0]
    
    if len(result) > max_length:
        return result[:max_length-3] + "..."
    
    return result + "."


def benchmark_model(model, image_path, num_runs=5):
    """
    Benchmark model performance.
    
    Args:
        model: VLM model instance
        image_path (str): Path to test image
        num_runs (int): Number of runs for benchmarking
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    times = []
    
    print(f"Running benchmark ({num_runs} runs)...")
    for i in range(num_runs):
        start = time.time()
        _ = model.describe_image(image_path)
        end = time.time()
        times.append(end - start)
        print(f"  Run {i+1}: {times[-1]:.2f}s")
    
    results = {
        "num_runs": num_runs,
        "times": times,
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times)
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Average: {results['mean_time']:.2f}s")
    print(f"  Min: {results['min_time']:.2f}s")
    print(f"  Max: {results['max_time']:.2f}s")
    
    return results


def download_sample_image(url, save_path="sample_image.jpg"):
    """
    Download a sample image from URL.
    
    Args:
        url (str): Image URL
        save_path (str): Path to save image
        
    Returns:
        str: Path to saved image
    """
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded image to: {save_path}")
        return save_path
    except ImportError:
        print("Error: requests library not installed. Install with: pip install requests")
        return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


if __name__ == "__main__":
    # Test utilities
    print("EJB VLM Utilities - Eduardo J. Barrios")
    print_device_info()
