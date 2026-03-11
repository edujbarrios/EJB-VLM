"""
CLIP-GPT2 Vision-Language Model (Parameterized)
Combines CLIP for image encoding and GPT-2 for text generation.

Author: Eduardo J. Barrios (@edujbarrIos)
"""

import torch
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import clip

from ..utils.config_loader import (
    get_model_config, 
    get_generation_params,
    get_categories,
    get_templates
)
from ..utils.device_utils import get_device


class EJBVLMDescriptor:
    """
    EJB Vision-Language Model (EJB-VLM) Descriptor.
    
    A zero-shot vision-language model that uses CLIP to encode images and GPT-2 
    to generate descriptions. This is a zero-shot approach that doesn't require 
    training the large models, instead using prompting techniques.
    
    Author: Eduardo J. Barrios (@edujbarrIos)
    """
    
    def __init__(self, config=None, clip_model_name=None, gpt_model_name=None, device=None):
        """
        Initialize the CLIP-GPT2 descriptor.
        
        Args:
            config (dict): Configuration dictionary (if None, loads from config.yaml)
            clip_model_name (str): CLIP model variant (overrides config)
            gpt_model_name (str): GPT-2 model variant (overrides config)
            device (str/torch.device): Device to run models on (overrides config)
        """
        # Load config
        if config is None:
            config = get_model_config()
        
        # Get model parameters from config or arguments
        self.clip_model_name = clip_model_name or config["clip"]["model_name"]
        self.gpt_model_name = gpt_model_name or config["gpt"]["model_name"]
        
        # Get device
        if device is not None:
            if isinstance(device, str):
                self.device = get_device(device)
            else:
                self.device = device
        else:
            self.device = get_device(config["clip"]["device"])
        
        print(f"Using device: {self.device}")
        
        # Load CLIP
        print(f"Loading CLIP ({self.clip_model_name})...")
        self.clip_model, self.clip_preprocess = clip.load(self.clip_model_name, device=self.device)
        self.clip_model.eval()
        
        # Load GPT-2
        print(f"Loading GPT-2 ({self.gpt_model_name})...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model_name)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(self.gpt_model_name).to(self.device)
        self.gpt_model.eval()
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load generation parameters
        self.default_gen_params = get_generation_params("default")
        
        print("Models loaded successfully!")
        
    def encode_image(self, image_path):
        """
        Encode an image using CLIP.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            torch.Tensor: Image features from CLIP
        """
        image = Image.open(image_path).convert("RGB")
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def get_image_context(self, image_features, templates=None, top_k=5):
        """
        Get relevant text context for the image using CLIP's text-image similarity.
        
        Args:
            image_features (torch.Tensor): CLIP image features
            templates (list): Template strings (if None, uses config)
            top_k (int): Number of top descriptions to consider
            
        Returns:
            str: Context string to prompt GPT-2
        """
        # Get templates from config if not provided
        if templates is None:
            templates = get_templates("basic")
        
        # Get text features for templates
        text_inputs = torch.cat([clip.tokenize(f"{t}") for t in templates]).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (image_features @ text_features.T).squeeze(0)
        top_idx = similarity.argsort(descending=True)[:top_k]
        
        # Build context
        best_template = templates[top_idx[0]]
        return best_template
    
    def describe_image(self, image_path, preset=None, **kwargs):
        """
        Generate a description for an image.
        
        Args:
            image_path (str): Path to image file
            preset (str): Generation preset (default, creative, focused, detailed)
            **kwargs: Override specific generation parameters
            
        Returns:
            str or list: Generated description(s)
        """
        # Get generation parameters
        if preset:
            gen_params = get_generation_params(preset)
        else:
            gen_params = self.default_gen_params.copy()
        
        # Override with kwargs
        gen_params.update(kwargs)
        
        # Encode image
        image_features = self.encode_image(image_path)
        
        # Get context prompt
        context = self.get_image_context(image_features)
        prompt = f"Image description: {context}"
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            output = self.gpt_model.generate(
                input_ids,
                max_length=gen_params.get("max_length", 50),
                num_return_sequences=gen_params.get("num_return_sequences", 1),
                temperature=gen_params.get("temperature", 0.7),
                top_p=gen_params.get("top_p", 0.9),
                top_k=gen_params.get("top_k", 50),
                num_beams=gen_params.get("num_beams", 5),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=gen_params.get("no_repeat_ngram_size", 2),
                early_stopping=gen_params.get("early_stopping", True)
            )
        
        # Decode generated text
        descriptions = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
        
        num_sequences = gen_params.get("num_return_sequences", 1)
        return descriptions[0] if num_sequences == 1 else descriptions
    
    def batch_describe_images(self, image_paths, **kwargs):
        """
        Generate descriptions for multiple images.
        
        Args:
            image_paths (list): List of image paths
            **kwargs: Additional arguments for describe_image
            
        Returns:
            dict: Dictionary mapping image paths to descriptions
        """
        results = {}
        for image_path in image_paths:
            print(f"Processing: {image_path}")
            try:
                description = self.describe_image(image_path, **kwargs)
                results[image_path] = description
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[image_path] = f"Error: {str(e)}"
        
        return results
    
    def compare_images(self, image_path1, image_path2):
        """
        Compare two images using CLIP embeddings.
        
        Args:
            image_path1 (str): Path to first image
            image_path2 (str): Path to second image
            
        Returns:
            float: Cosine similarity between images (0-1)
        """
        features1 = self.encode_image(image_path1)
        features2 = self.encode_image(image_path2)
        
        similarity = (features1 @ features2.T).item()
        return similarity


class AdvancedEJBVLMDescriptor(EJBVLMDescriptor):
    """
    Advanced EJB-VLM Descriptor with enhanced features:
    - Category detection using CLIP's zero-shot capabilities
    - Multiple description generation
    - Image comparison and similarity scoring
    
    Author: Eduardo J. Barrios (@edujbarrIos)
    """
    
    def __init__(self, category_set="default", *args, **kwargs):
        """
        Initialize advanced EJB-VLM descriptor.
        
        Args:
            category_set (str): Which category set to use (default or extended)
        """
        super().__init__(*args, **kwargs)
        
        # Load categories from config
        self.categories = get_categories(category_set)
    
    def detect_categories(self, image_path, top_k=3):
        """
        Detect image categories using CLIP zero-shot classification.
        
        Args:
            image_path (str): Path to image file
            top_k (int): Number of top categories to return
            
        Returns:
            list: List of (category, confidence) tuples
        """
        image_features = self.encode_image(image_path)
        
        # Create text prompts for categories
        text_prompts = [f"a photo of a {cat}" for cat in self.categories]
        text_inputs = torch.cat([clip.tokenize(t) for t in text_prompts]).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity and softmax
            similarity = (image_features @ text_features.T).squeeze(0)
            probs = similarity.softmax(dim=0)
        
        # Get top-k categories
        top_probs, top_indices = probs.topk(top_k)
        results = [(self.categories[idx], prob.item()) for idx, prob in zip(top_indices, top_probs)]
        
        return results
    
    def detailed_description(self, image_path, num_descriptions=3, preset=None):
        """
        Generate a detailed description combining category detection and text generation.
        
        Args:
            image_path (str): Path to image file
            num_descriptions (int): Number of alternative descriptions
            preset (str): Generation preset to use
            
        Returns:
            dict: Dictionary with categories and descriptions
        """
        # Detect categories
        categories = self.detect_categories(image_path)
        
        # Generate descriptions
        descriptions = self.describe_image(
            image_path,
            preset=preset,
            num_return_sequences=num_descriptions,
            max_length=60
        )
        
        return {
            "categories": categories,
            "descriptions": descriptions if isinstance(descriptions, list) else [descriptions]
        }
