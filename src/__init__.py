"""
EJB-VLM (Eduardo J. Barrios Vision-Language Model)
Author: Eduardo J. Barrios (@edujbarrIos)
"""

__version__ = "0.1.0"
__author__ = "Eduardo J. Barrios"

from .models.ejb_vlm_model import EJBVLMDescriptor, AdvancedEJBVLMDescriptor
from .utils.config_loader import load_config, get_config

__all__ = [
    "EJBVLMDescriptor",
    "AdvancedEJBVLMDescriptor",
    "load_config",
    "get_config"
]
