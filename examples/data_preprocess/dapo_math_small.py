#!/usr/bin/env python3
"""
Create a small subset of DAPO-Math-17k dataset for testing.
"""

import json
import os
from datasets import load_dataset

def create_small_dataset():
    print("Loading DAPO-Math-17k dataset...")
    
    # Load dataset
    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")
    
    # Create output directory
    output_dir = "/data2/lixy/VerlCoder/data/dapo_math_17k"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process small subset for testing (first 1000 examples)
    print("Processing small subset...")
    small_data = []
    for i, example in enumerate(dataset["train"]):
        if i >= 1000:  # Only take first 1000 examples
            break
        
        # Extract the user prompt content
        user_content = example["prompt"][0]["content"]
        # Convert to format expected by VERL
        processed = {
            "conversations": [
                {
                    "role": "user", 
                    "content": user_content
                }
            ]
        }
        small_data.append(processed)
    
    # Save small training data
    train_file = os.path.join(output_dir, "train_small.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in small_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(small_data)} training examples to {train_file}")
    
    # Create even smaller subset for quick testing (100 examples)
    test_data = small_data[:100]
    test_file = os.path.join(output_dir, "test_tiny.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(test_data)} test examples to {test_file}")
    print("Small dataset creation completed!")

if __name__ == "__main__":
    create_small_dataset()