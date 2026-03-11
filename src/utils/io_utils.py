"""
I/O utility functions for EJB VLM
Author: Eduardo J. Barrios (@edujbarruos)
"""

import json
import time
from typing import Dict, Any


def save_results_to_json(results: Dict[str, Any], output_path: str = "results.json"):
    """
    Save description results to a JSON file.
    
    Args:
        results (dict): Dictionary of results
        output_path (str): Path to output JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


def format_description(description: str, max_length: int = 100) -> str:
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


def benchmark_model(model, image_path: str, num_runs: int = 5) -> Dict[str, Any]:
    """
    Benchmark model performance.
    
    Args:
        model: VLM model instance
        image_path (str): Path to test image
        num_runs (int): Number of runs for benchmarking
        
    Returns:
        dict: Benchmark results
    """
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
