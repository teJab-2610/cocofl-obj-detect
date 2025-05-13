#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

def load_pkl_file(file_path):
    """Load data from a pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def average_client_losses(losses):
    """Average the 5 loss values for each round to get 50 avg loss values"""
    return [np.mean(round_losses) for round_losses in losses]

def extract_map_values(map_data):
    """Extract the np.float64 values from the map data"""
    return [float(item[0]) for item in map_data]

def plot_metrics(data, output_path=None):
    """Plot the requested metrics from the pickle data"""
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Fed Avg on Soda10m IID distribution', fontsize=16)
    
    # X-axis data for all plots (rounds 1-50)
    rounds = list(range(1, 51))
    
    # Map client IDs to simple client numbers for display
    client_id_map = {
        'ssl40': 'Client 1',
        'ssl41': 'Client 2', 
        'ssl42': 'Client 3'
    }
    
    # 1. Client Losses Plot
    ax1.set_title('Clients Loss')
    ax1.set_xlabel('Rounds')
    ax1.set_ylabel('Loss')
    
    # Process and plot client loss data for each client
    client_ids = ['ssl40', 'ssl41', 'ssl42']
    for client_id in client_ids:
        loss_key = f'client_{client_id}_train_loss'
        if loss_key in data:
            losses = data[loss_key]
            avg_losses = average_client_losses(losses)
            ax1.plot(rounds, avg_losses, label=client_id_map[client_id])
    
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Client Metrics Plot
    ax2.set_title('Clients Metrics')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel('mAP / mAP50')
    
    # Plot mAP and mAP50 for each client
    for client_id in client_ids:
        map_key = f'client_{client_id}_val_map'
        map50_key = f'client_{client_id}_val_map50'
        
        if map_key in data:
            map_values = extract_map_values(data[map_key])
            ax2.plot(rounds, map_values, label=f'{client_id_map[client_id]} mAP')
        
        if map50_key in data:
            map50_values = extract_map_values(data[map50_key])
            ax2.plot(rounds, map50_values, linestyle='--', label=f'{client_id_map[client_id]} mAP50')
    
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Global Metrics Plot
    ax3.set_title('Global Metrics')
    ax3.set_xlabel('Rounds')
    ax3.set_ylabel('Metric Value')
    
    # Plot global metrics (ignore the first value as per instructions)
    if 'global_mAP' in data:
        # Skip first value if there are 51 values, otherwise use as is
        global_map = data['global_mAP'][1:51] if len(data['global_mAP']) == 51 else data['global_mAP']
        ax3.plot(rounds, global_map, label='Global mAP')
    
    if 'global_mAP50' in data:
        # Skip first value if there are 51 values, otherwise use as is
        global_map50 = data['global_mAP50'][1:51] if len(data['global_mAP50']) == 51 else data['global_mAP50']
        ax3.plot(rounds, global_map50, label='Global mAP50')
    
    if 'global_precision' in data:
        # Skip first value if there are 51 values, otherwise use as is
        global_precision = data['global_precision'][1:51] if len(data['global_precision']) == 51 else data['global_precision']
        ax3.plot(rounds, global_precision, label='Global Precision')
    
    if 'global_recall' in data:
        # Skip first value if there are 51 values, otherwise use as is
        global_recall = data['global_recall'][1:51] if len(data['global_recall']) == 51 else data['global_recall']
        ax3.plot(rounds, global_recall, label='Global Recall')
    
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Add dataset sizes as text in the top-right corner of the figure
    dataset_sizes_text = ""
    for i, client_id in enumerate(client_ids, 1):
        dataset_size_key = f'client_{client_id}_dataset_size'
        if dataset_size_key in data:
            dataset_sizes_text += f"Client {i} Dataset Size: {data[dataset_size_key]}\n"
    
    if dataset_sizes_text:
        # Position text in top-right corner with some padding
        plt.figtext(0.98, 0.98, dataset_sizes_text, ha='right', va='top', fontsize=10, 
                   bbox={"facecolor":"lightgray", "alpha":0.7, "pad":5})
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Extract and plot metrics from a pickle file")
    parser.add_argument("file", help="Path to the pickle file")
    parser.add_argument("--output", "-o", help="Path to save the output plot")
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
    
    # Plot the metrics
    plot_metrics(data, args.output)

if __name__ == "__main__":
    main()