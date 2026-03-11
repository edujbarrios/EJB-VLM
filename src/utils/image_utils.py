"""
Image utility functions for EJB VLM
Author: Eduardo J. Barrios (@edujbarruos)
"""

import os
from pathlib import Path
from PIL import Image
from typing import List


def validate_image_path(image_path: str) -> bool:
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


def load_images_from_directory(directory: str, recursive: bool = False) -> List[str]:
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


def resize_image(image_path: str, max_size: tuple = (800, 800), output_path: str = None) -> str:
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


def create_image_grid(image_paths: List[str], grid_size: tuple = (2, 2), 
                     output_path: str = "grid.jpg") -> str:
    """
    Create a grid of images.
    
    Args:
        image_paths (list): List of image paths
        grid_size (tuple): (rows, cols) for grid
        output_path (str): Path to save grid
        
    Returns:
        str: Path to saved grid
    """
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


def download_sample_image(url: str, save_path: str = "sample_image.jpg") -> str:
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
