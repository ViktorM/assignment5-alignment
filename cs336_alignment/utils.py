import os
from pathlib import Path


def get_model_path():
    """Get model path based on environment."""
    # Check if we're on the cluster (usually has a specific env var or path)
    cluster_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    local_path = "/home/viktor4090/Projects/Stanford/CS336/assignment5-alignment/cs336_alignment/data/models/Qwen2.5-Math-1.5B"

    # Option 1: Use environment variable
    if model_path := os.environ.get("QWEN_MODEL_PATH"):
        return model_path

    # Option 2: Check which path exists
    if Path(cluster_path).exists():
        return cluster_path
    elif Path(local_path).exists():
        return local_path
    else:
        raise ValueError(f"Model not found in cluster path ({cluster_path}) or local path ({local_path})")