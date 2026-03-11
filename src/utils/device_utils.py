"""
Device utility functions for EJB VLM
Author: Eduardo J. Barrios (@edujbarrIos)
"""

import torch
from typing import Dict


def get_device_info() -> Dict[str, any]:
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


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get torch device based on string specification.
    
    Args:
        device_str (str): Device specification ("auto", "cuda", "cpu")
        
    Returns:
        torch.device: Torch device
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)
