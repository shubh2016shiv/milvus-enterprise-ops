"""
Common Utilities for Milvus Usage Examples

Provides helper functions for generating test data, formatting output,
and managing test collections across all usage examples.
"""

import numpy as np
import random
import string
from typing import List, Dict, Any
from pymilvus import utility, connections


def generate_random_vectors(count: int, dimension: int) -> List[List[float]]:
    """
    Generate random normalized vectors for testing.
    
    Args:
        count: Number of vectors to generate
        dimension: Dimension of each vector
        
    Returns:
        List of random vectors
    """
    vectors = np.random.rand(count, dimension).astype(np.float32)
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors.tolist()


def generate_random_text() -> str:
    """Generate random text for metadata fields."""
    words = ["quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
             "cat", "mouse", "bird", "fish", "tree", "flower", "sun"]
    return " ".join(random.choices(words, k=random.randint(3, 8)))


def generate_test_data(count: int, dimension: int, include_metadata: bool = True) -> Dict[str, List]:
    """
    Generate test data for insertion.
    
    Args:
        count: Number of entities to generate
        dimension: Vector dimension
        include_metadata: Whether to include metadata fields
        
    Returns:
        Dictionary with field names as keys and lists of values
    """
    data = {
        "id": list(range(1, count + 1)),
        "vector": generate_random_vectors(count, dimension)
    }
    
    if include_metadata:
        data["text"] = [generate_random_text() for _ in range(count)]
        data["value"] = [random.randint(1, 100) for _ in range(count)]
        data["category"] = [random.choice(["A", "B", "C", "D"]) for _ in range(count)]
    
    return data


def print_section(title: str, width: int = 60):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of the header line
    """
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_step(step_num: int, description: str):
    """
    Print a step description.
    
    Args:
        step_num: Step number
        description: Step description
    """
    print(f"\n[Step {step_num}] {description}")


def print_success(message: str):
    """Print a success message."""
    print(f"[SUCCESS] {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"[ERROR] {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"[WARNING] {message}")
    
def print_note(message: str):
    """Print an informational note."""
    print(f"[NOTE] {message}")


def print_info(key: str, value: Any):
    """Print information in key-value format."""
    print(f"  • {key}: {value}")


def cleanup_collection(collection_name: str, conn_alias: str = "default"):
    """
    Clean up a test collection if it exists.
    
    Args:
        collection_name: Name of collection to clean up
        conn_alias: Connection alias
    """
    try:
        if utility.has_collection(collection_name, using=conn_alias):
            utility.drop_collection(collection_name, using=conn_alias)
            print_success(f"Cleaned up collection: {collection_name}")
        else:
            print_info("Status", f"Collection '{collection_name}' does not exist")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")


def get_connection_config() -> Dict[str, Any]:
    """
    Get default connection configuration.
    
    Returns:
        Dictionary with connection parameters
    """
    return {
        "host": "localhost",
        "port": "19530",
        "timeout": 30
    }


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def print_results_table(results: List[Dict[str, Any]], max_rows: int = 10):
    """
    Print search results in a formatted table.
    
    Args:
        results: List of result dictionaries
        max_rows: Maximum number of rows to display
    """
    if not results:
        print("No results to display")
        return
    
    print("\nSearch Results:")
    print("-" * 80)
    print(f"{'ID':<10} {'Score':<15} {'Details':<50}")
    print("-" * 80)
    
    for i, result in enumerate(results[:max_rows]):
        result_id = result.get('id', 'N/A')
        score = result.get('distance', result.get('score', 'N/A'))
        
        # Extract some metadata for display
        details = []
        for key, value in result.items():
            if key not in ['id', 'distance', 'score', 'vector']:
                details.append(f"{key}={value}")
        details_str = ", ".join(details[:2]) if details else "No metadata"
        
        print(f"{str(result_id):<10} {score:<15.4f} {details_str:<50}")
    
    if len(results) > max_rows:
        print(f"... and {len(results) - max_rows} more results")
    print("-" * 80)


