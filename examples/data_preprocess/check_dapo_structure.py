#!/usr/bin/env python3
"""
Check DAPO-Math-17k dataset structure.
"""

from datasets import load_dataset

def check_dataset_structure():
    print("Loading DAPO-Math-17k dataset...")
    
    # Load dataset
    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")
    
    print("Dataset keys:", dataset.keys())
    
    if "train" in dataset:
        print("\nTrain split info:")
        print(f"Number of examples: {len(dataset['train'])}")
        
        # Check first example
        first_example = dataset["train"][0]
        print(f"\nFirst example keys: {first_example.keys()}")
        print(f"First example: {first_example}")
        
        # Check a few more examples to understand structure
        for i in range(min(3, len(dataset["train"]))):
            example = dataset["train"][i]
            print(f"\nExample {i} keys: {example.keys()}")

if __name__ == "__main__":
    check_dataset_structure()