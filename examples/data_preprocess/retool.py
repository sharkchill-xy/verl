"""
Preprocess the ReTool-SFT dataset to parquet format
"""

import argparse
import os
import re

import datasets
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs


def extract_boxed_answer(response_str):
    """从ReTool回答中提取最终答案"""
    # 寻找 \boxed{答案} 格式
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(boxed_pattern, response_str)
    if match:
        return match.group(1)
    
    # 如果没找到boxed格式，尝试从<answer>标签中提取
    answer_pattern = r'<answer>\s*\\boxed\{([^}]+)\}\s*</answer>'
    match = re.search(answer_pattern, response_str)
    if match:
        return match.group(1)
    
    # 如果都没找到，返回None
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data2/lixy/VerlCoder/data/retool")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_length", type=int, default=8192, help="过滤超过此长度的样本")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-7B-Instruct", help="用于长度过滤的tokenizer")
    parser.add_argument("--no_filter", action="store_true", help="跳过长度过滤")

    args = parser.parse_args()

    data_source = "JoeYing/ReTool-SFT"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    
    # 创建验证集（从训练集中分割10%）
    dataset_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    # 可选的长度过滤
    if not args.no_filter:
        print(f"加载 {args.tokenizer_name} tokenizer 进行长度过滤...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    def make_map_fn(split):
        def process_fn(example, idx):
            messages = example["messages"]
            
            # 确保是对话格式（用户+助手）
            if len(messages) != 2:
                return None
                
            user_message = messages[0]
            assistant_message = messages[1]
            
            if user_message["role"] != "user" or assistant_message["role"] != "assistant":
                return None
                
            prompt = user_message["content"]
            response = assistant_message["content"]
            
            # 长度过滤（如果启用）
            if not args.no_filter:
                # 使用tokenizer的chat template来计算总长度
                conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                formatted_text = tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
                total_length = len(tokens)
                
                if total_length > args.max_length:
                    return None  # 过滤掉超长样本
            
            # 提取最终答案作为ground truth（如果有的话）
            ground_truth = extract_boxed_answer(response)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "reasoning_with_tool",
                "reward_model": {"style": "rule", "ground_truth": ground_truth} if ground_truth else {"style": "preference"},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "response": response,
                    "prompt_text": prompt,
                },
            }
            return data

        return process_fn

    # 过滤掉None值（无效数据）
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)
    val_dataset = val_dataset.filter(lambda x: x is not None)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    hdfs_dir = args.hdfs_dir

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))

    print(f"数据已保存到: {local_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"数据已复制到HDFS: {hdfs_dir}")