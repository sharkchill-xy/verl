#!/usr/bin/env python3
"""
Convert DAPO Math data to VERL RL training format.
"""

import json
import pandas as pd
from pathlib import Path
import argparse


def convert_dapo_to_rl_format(input_file: str, output_file: str, max_samples: int = None):
    """Convert DAPO Math data to VERL RL format."""
    print(f"Converting {input_file} to RL format...")
    
    # Read DAPO data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Read {len(data)} samples from DAPO dataset")
    
    if max_samples:
        data = data[:max_samples]
        print(f"Limited to {len(data)} samples")
    
    # Convert to RL format
    rl_data = []
    for i, item in enumerate(data):
        # Extract prompt from conversations
        conversations = item.get('conversations', [])
        if not conversations:
            continue
            
        # Find user message
        user_content = None
        for conv in conversations:
            if conv.get('role') == 'user':
                user_content = conv.get('content')
                break
        
        if not user_content:
            continue
            
        # Create RL format entry
        rl_entry = {
            'data_source': 'dapo-math',
            'prompt': [{'content': user_content, 'role': 'user'}],
            'ability': 'math',
            'reward_model': {'ground_truth': '', 'style': 'rule'},  # Will be filled by reward function
            'extra_info': {
                'index': i,
                'original_conversations': conversations
            }
        }
        
        rl_data.append(rl_entry)
    
    print(f"Converted {len(rl_data)} samples to RL format")
    
    # Save as parquet
    df = pd.DataFrame(rl_data)
    df.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert DAPO Math data to VERL RL format')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output Parquet file')
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to convert')
    
    args = parser.parse_args()
    
    convert_dapo_to_rl_format(args.input, args.output, args.max_samples)


if __name__ == "__main__":
    # Convert test data for quick testing
    convert_dapo_to_rl_format(
        input_file="/data2/lixy/VerlCoder/data/dapo_math_17k/test_tiny.jsonl",
        output_file="/data2/lixy/VerlCoder/data/dapo_math_17k/test_tiny_rl.parquet",
        max_samples=50  # Small size for testing
    )
    
    # Convert training data
    convert_dapo_to_rl_format(
        input_file="/data2/lixy/VerlCoder/data/dapo_math_17k/train_small.jsonl", 
        output_file="/data2/lixy/VerlCoder/data/dapo_math_17k/train_small_rl.parquet",
        max_samples=200  # Medium size for training
    )