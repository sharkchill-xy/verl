#!/usr/bin/env python3
"""
Split DAPO dataset into train/val splits
"""
import datasets
import os

def split_dataset():
    # 读取原始数据
    input_file = "/data2/lixy/VerlCoder/data/retool_dapo/train.parquet"
    output_dir = "/data2/lixy/VerlCoder/data/retool_dapo_split"
    
    print(f"Reading dataset from {input_file}")
    dataset = datasets.load_dataset("parquet", data_files=input_file)["train"]
    print(f"Total samples: {len(dataset)}")
    
    # 分割数据：val=1000，剩余为train
    val_size = 1000
    train_size = len(dataset) - val_size
    
    print(f"Splitting into train={train_size}, val={val_size}")
    
    # 使用datasets的train_test_split方法
    split_dataset = dataset.train_test_split(
        test_size=val_size, 
        seed=42,
        shuffle=True
    )
    
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]  # datasets将test_size部分命名为"test"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分割后的数据
    train_output = os.path.join(output_dir, "train.parquet")
    val_output = os.path.join(output_dir, "val.parquet")
    
    train_dataset.to_parquet(train_output)
    val_dataset.to_parquet(val_output)
    
    print(f"Train set saved to: {train_output} ({len(train_dataset)} samples)")
    print(f"Val set saved to: {val_output} ({len(val_dataset)} samples)")
    
    return train_output, val_output

if __name__ == "__main__":
    split_dataset()