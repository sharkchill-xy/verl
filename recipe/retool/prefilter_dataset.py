#!/usr/bin/env python3
"""
预过滤ReTool训练数据集
根据max_prompt_length和max_response_length过滤DAPO数据集，避免每次训练时都要等待数据过滤。
"""

import argparse
import os
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import time

def main():
    parser = argparse.ArgumentParser(description="预过滤ReTool训练数据集")
    parser.add_argument("--input_file", type=str, required=True, help="输入的parquet文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出的过滤后parquet文件路径")
    parser.add_argument("--tokenizer_path", type=str, 
                       default="/data2/lixy/VerlCoder/checkpoints/retool-multiturn-sft/retool-multiturn-sft-qwen2.5-7b-sp4-lr5e-6/global_step_42",
                       help="tokenizer路径")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="最大prompt长度")
    parser.add_argument("--max_response_length", type=int, default=4096, help="最大response长度")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="prompt字段名")
    parser.add_argument("--num_workers", type=int, default=8, help="并行处理workers数量")
    
    args = parser.parse_args()
    
    print(f"开始预过滤数据集...")
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    print(f"最大prompt长度: {args.max_prompt_length}")
    print(f"最大response长度: {args.max_response_length}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # 加载数据集
    print("加载数据集...")
    dataset = datasets.load_dataset("parquet", data_files=args.input_file)["train"]
    print(f"原始数据集大小: {len(dataset):,}")
    
    # 定义过滤函数
    def doc2len(doc) -> dict:
        """计算prompt和response的token长度"""
        try:
            # 计算prompt长度
            prompt_messages = doc[args.prompt_key]
            prompt_tokens = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
            prompt_length = len(prompt_tokens)
            
            # 如果有response，计算response长度（这里假设response在某个字段中）
            # 由于我们主要关心prompt长度，先只过滤prompt
            response_length = 0  # 暂时设为0，可以根据实际数据结构调整
            
            return {
                "prompt_length": prompt_length,
                "response_length": response_length,
                "valid": prompt_length <= args.max_prompt_length
            }
        except Exception as e:
            print(f"处理样本时出错: {e}")
            return {
                "prompt_length": float('inf'),
                "response_length": float('inf'), 
                "valid": False
            }
    
    # 定义真正的过滤函数
    def filter_fn(doc) -> bool:
        """过滤函数：判断样本是否符合长度要求"""
        try:
            prompt_messages = doc[args.prompt_key]
            prompt_tokens = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
            prompt_length = len(prompt_tokens)
            return prompt_length <= args.max_prompt_length
        except Exception as e:
            print(f"过滤样本时出错: {e}")
            return False
    
    # 执行过滤
    print(f"开始过滤长度超过 {args.max_prompt_length} tokens的prompts...")
    start_time = time.time()
    
    filtered_dataset = dataset.filter(
        filter_fn,
        num_proc=args.num_workers,
        desc=f"Filtering prompts longer than {args.max_prompt_length} tokens"
    )
    
    end_time = time.time()
    print(f"过滤完成，耗时: {end_time - start_time:.2f}秒")
    print(f"过滤后数据集大小: {len(filtered_dataset):,}")
    print(f"过滤掉的样本数: {len(dataset) - len(filtered_dataset):,}")
    print(f"保留比例: {len(filtered_dataset) / len(dataset) * 100:.2f}%")
    
    # 保存过滤后的数据集
    print(f"保存过滤后的数据集到: {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    filtered_dataset.to_parquet(args.output_file)
    
    print("✅ 预过滤完成！")

if __name__ == "__main__":
    main()