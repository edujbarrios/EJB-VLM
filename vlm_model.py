"""
CLIP-GPT2 Vision-Language Model
Combines CLIP for image encoding and GPT-2 for text generation.

Author: Eduardo J. Barrios (@edujbarruos)
"""

import torch
import torch.nn as nn
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import clip
import numpy as np


class CLIPGPTDescriptor:
    """
    A vision-language model that uses CLIP to encode images and GPT-2 to generate descriptions.
    
    This is a zero-shot approach that doesn't require training the large models,
    instead using prompting and prefix tuning techniques.
    """
    
    def __init__(self, clip_model_name="ViT-B/32", gpt_model_name="gpt2", device=None):
        """
        Initialize the CLIP-GPT2 descriptor.
        
        Args:
            clip_model_name (str): CLIP model variant to use
            gpt_model_name (str): GPT-2 model variant to use
            device (torch.device): Device to run models on
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP
        print("Loading CLIP...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()
        
        # Load GPT-2
        print("Loading GPT-2...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name).to(self.device)
        self.gpt_model.eval()
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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
    
    def get_image_context(self, image_features, top_k=5):
        """
        Get relevant text context for the image using CLIP's text-image similarity.
        
        Args:
            image_features (torch.Tensor): CLIP image features
            top_k (int): Number of top descriptions to consider
            
        Returns:
            str: Context string to prompt GPT-2
        """
        # Predefined image description templates
        templates = [
            "a photo of",
            "an image showing",
            "a picture of",
            "this is",
            "there is",
            "the image contains",
            "showing",
            "depicting",
            "featuring",
            "with",
        ]
        
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
    
    def describe_image(self, image_path, max_length=50, num_return_sequences=1, 
                      temperature=0.7, top_p=0.9, num_beams=5):
        """
        Generate a description for an image.
        
        Args:
            image_path (str): Path to image file
            max_length (int): Maximum length of generated text
            num_return_sequences (int): Number of descriptions to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            num_beams (int): Number of beams for beam search
            
        Returns:
            str or list: Generated description(s)
        """
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
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode generated text
        descriptions = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
        
        return descriptions[0] if num_return_sequences == 1 else descriptions
    
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


class AdvancedCLIPGPTDescriptor(CLIPGPTDescriptor):
    """
    Extended version with additional capabilities like category detection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Common categories for zero-shot classification
        self.categories = [
            "person", "animal", "vehicle", "building", "nature",
            "food", "indoor scene", "outdoor scene", "object",
            "sports", "technology", "art"
        ]
    
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
    
    def detailed_description(self, image_path, num_descriptions=3):
        """
        Generate a detailed description combining category detection and text generation.
        
        Args:
            image_path (str): Path to image file
            num_descriptions (int): Number of alternative descriptions
            
        Returns:
            dict: Dictionary with categories and descriptions
        """
        # Detect categories
        categories = self.detect_categories(image_path)
        
        # Generate descriptions
        descriptions = self.describe_image(
            image_path, 
            num_return_sequences=num_descriptions,
            max_length=60
        )
        
        return {
            "categories": categories,
            "descriptions": descriptions if isinstance(descriptions, list) else [descriptions]
        }
