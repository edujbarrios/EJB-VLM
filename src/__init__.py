"""
EJB Vision-Language Model Package
Author: Eduardo J. Barrios (@edujbarruos)
"""

__version__ = "0.1.0"
__author__ = "Eduardo J. Barrios"

from .models.ejb_vlm_model import CLIPGPTDescriptor, AdvancedCLIPGPTDescriptor
from .utils.config_loader import load_config, get_config

__all__ = [
    "CLIPGPTDescriptor",
    "AdvancedCLIPGPTDescriptor",
    "load_config",
    "get_config"
]
