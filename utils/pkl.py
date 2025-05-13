import pickle
import argparse
import pprint
import os
import sys
from collections import defaultdict

def load_pkl_file(file_path):
    """Load data from a pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def analyze_structure(data, prefix="", max_depth=3, current_depth=0):
    """Recursively analyze the structure of nested data"""
    if current_depth >= max_depth:
        return f"{type(data).__name__} (max depth reached)"
    
    if isinstance(data, dict):
        result = "{\n"
        for k, v in data.items():
            result += f"{prefix}  '{k}': {analyze_structure(v, prefix + '  ', max_depth, current_depth + 1)},\n"
        result += prefix + "}"
        return result
    elif isinstance(data, list):
        if not data:
            return "[]"
        if len(data) > 5:
            return f"[... {len(data)} items of type {type(data[0]).__name__} ...]"
        result = "[\n"
        for item in data:
            result += f"{prefix}  {analyze_structure(item, prefix + '  ', max_depth, current_depth + 1)},\n"
        result += prefix + "]"
        return result
    elif isinstance(data, (int, float, str, bool, type(None))):
        return repr(data)
    else:
        return f"{type(data).__name__}"

def print_metrics_info(metrics, detail_level=1):
    """Print information about the metrics based on detail level"""
    if not metrics:
        print("No metrics found in the file.")
        return
    
    # Basic information
    print(f"\n=== Metrics Overview ===")
    print(f"Number of metric keys: {len(metrics)}")
    
    # Print keys and their values
    print("\n=== Metric Keys and Values ===")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, list):
            if len(value) > 0:
                # Print the first value from the list as a sample
                print(f"{key}: list with {len(value)} items, first value: {value[0]}")
            else:
                print(f"{key}: empty list")
        elif isinstance(value, dict):
            dict_size = len(value)
            # Print the first key-value pair from the dict as a sample
            if dict_size > 0:
                first_key = next(iter(value))
                print(f"{key}: dict with {dict_size} items, sample: {first_key} -> {value[first_key]}")
            else:
                print(f"{key}: empty dict")
        else:
            # For simple values, print directly
            print(f"{key}: {value}")
    
    # More detailed information if requested
    if detail_level >= 2:
        print("\n=== Detailed Structure ===")
        structure = analyze_structure(metrics)
        print(structure)
    
    # Full content if highest detail level
    if detail_level >= 3:
        print("\n=== Full Content ===")
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(metrics)

def main():
    parser = argparse.ArgumentParser(description="Read and print data from a pickle file")
    parser.add_argument("file", help="Path to the pickle file")
    parser.add_argument("--detail", "-d", type=int, choices=[1, 2, 3], default=1,
                        help="Detail level: 1=basic, 2=structure, 3=full content (default: 1)")
    parser.add_argument("--keys", "-k", nargs="+", help="Only show these specific keys")
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist.")
        sys.exit(1)
    
    # Load the pickle file
    print(f"Loading pickle file: {args.file}")
    data = load_pkl_file(args.file)
    
    if data is None:
        print("Failed to load the pickle file.")
        sys.exit(1)
    
    # Filter to specific keys if requested
    if args.keys:
        if not isinstance(data, dict):
            print("Cannot filter keys - data is not a dictionary.")
        else:
            filtered_data = {}
            for key in args.keys:
                if key in data:
                    filtered_data[key] = data[key]
                else:
                    print(f"Warning: Key '{key}' not found in the data.")
            data = filtered_data
    
    # Print the data with the specified detail level
    print_metrics_info(data, args.detail)
    
    # Analyze client-specific metrics if the data looks like federated learning metrics
    client_metrics = defaultdict(list)
    if isinstance(data, dict):
        for key in data.keys():
            if key.startswith('client_'):
                parts = key.split('_')
                if len(parts) >= 3:
                    client_id = '_'.join(parts[1:-1])  # Handle client IDs with underscores
                    metric_name = parts[-1]
                    client_metrics[client_id].append(metric_name)
    
    if client_metrics:
        print("\n=== Client-Specific Metrics ===")
        for client_id, metrics in client_metrics.items():
            print(f"Client {client_id}: {', '.join(sorted(metrics))}")
            # Also print the actual values for each client metric
            print("  Values:")
            for metric in sorted(metrics):
                key = f"client_{client_id}_{metric}"
                print(f"    {metric}: {data.get(key)}")

if __name__ == "__main__":
    main()