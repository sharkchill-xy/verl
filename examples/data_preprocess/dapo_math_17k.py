#!/usr/bin/env python3
"""
Download and preprocess DAPO-Math-17k dataset for ReTool RL training.
"""

import json
import os
from datasets import load_dataset

def download_and_process_dapo_math():
    print("Downloading DAPO-Math-17k dataset...")
    
    # Load dataset
    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")
    
    # Create output directory
    output_dir = "/data2/lixy/VerlCoder/data/dapo_math_17k"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train split
    train_data = []
    for example in dataset["train"]:
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
        train_data.append(processed)
    
    # Save train data
    train_file = os.path.join(output_dir, "train.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(train_data)} training examples to {train_file}")
    
    # Process validation split if exists
    if "validation" in dataset:
        val_data = []
        for example in dataset["validation"]:
            user_content = example["prompt"][0]["content"]
            processed = {
                "conversations": [
                    {
                        "role": "user", 
                        "content": user_content
                    }
                ]
            }
            val_data.append(processed)
        
        val_file = os.path.join(output_dir, "val.jsonl")
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(val_data)} validation examples to {val_file}")
    
    # Create small subset for testing
    test_data = train_data[:100]  # First 100 examples for quick testing
    test_file = os.path.join(output_dir, "test_small.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(test_data)} test examples to {test_file}")
    print("DAPO-Math-17k dataset processing completed!")

if __name__ == "__main__":
    download_and_process_dapo_math()